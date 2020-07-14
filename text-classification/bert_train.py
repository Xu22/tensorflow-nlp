import tensorflow as tf
import numpy as np
import os
import time
from utils import dataHelper as data_helpers
# import data_helpers
from model.bert_fune import Model, BertConfig
from bert import tokenization
from tqdm import tqdm
from sklearn.model_selection import train_test_split

config_parm = BertConfig()
tokenization.validate_case_matches_checkpoint(True, config_parm.BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(
      vocab_file=config_parm.BERT_VOCAB, do_lower_case=True)

print("Loading data...")

x_text, y, y_onehot = data_helpers.load_data_and_labels(config_parm, is_training=True)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))#np.arange生成随机序列
x_text = np.array(x_text)[shuffle_indices]
y = np.array(y)[shuffle_indices]

input_ids, input_masks, segment_ids = [], [], []

for text in tqdm(x_text):
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > config_parm.seq_len - 2:
        tokens_a = tokens_a[:(config_parm.seq_len - 2)]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_id = [0] * len(tokens)
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    padding = [0] * (config_parm.seq_len - len(input_id))
    input_id += padding
    input_mask += padding
    segment_id += padding

    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)

train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids, train_Y, test_Y = train_test_split(
    input_ids, input_masks, segment_ids, y, test_size = 0.2
)

# Training
def optimistic_restore(sess, save_file):
    """载入bert模型"""
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],tf.global_variables()),tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                # print("going to restore.var_name:",var_name,";saved_var_name:",saved_var_name)
                restore_vars.append(curr_var)
            else:
                print("variable not trained.var_name:",var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, save_file)

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
with tf.Session(config=config_gpu) as sess:
    model = Model(config_parm)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", config_parm.model_dir))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    optimistic_restore(sess, config_parm.BERT_INIT_CHKPNT)

    # 模型保存目录，如果没有则创建
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.acc)

    # Train Summaries
    # 训练数据保存
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    # 测试数据保存
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    current_step = 0
    EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

    while True:
        lasttime = time.time()
        if CURRENT_CHECKPOINT == EARLY_STOPPING:
            print('break epoch:%d\n' % (EPOCH))
            break

        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        pbar = tqdm(
            range(0, len(train_input_ids), config_parm.batch_size), desc='train minibatch loop'
        )
        current_step = 0
        for i in pbar:
            index = min(i + config_parm.batch_size, len(train_input_ids))
            batch_x = train_input_ids[i: index]
            batch_masks = train_input_masks[i: index]
            batch_segment = train_segment_ids[i: index]
            batch_y = train_Y[i: index]
            acc, loss, summary = model.train(sess, batch_x, batch_y, batch_segment, batch_masks)
            train_summary_writer.add_summary(summary, current_step)
            assert not np.isnan(loss)
            train_loss += loss
            train_acc += acc
            pbar.set_postfix(cost=loss, accuracy=acc)
            current_step += 1
        pbar = tqdm(range(0, len(test_input_ids), config_parm.batch_size), desc='test minibatch loop')
        current_step = 0
        for i in pbar:
            index = min(i + config_parm.batch_size, len(test_input_ids))
            batch_x = test_input_ids[i: index]
            batch_masks = test_input_masks[i: index]
            batch_segment = test_segment_ids[i: index]
            batch_y = test_Y[i: index]
            acc, loss, summary = model.eval(sess, batch_x, batch_y, batch_segment, batch_masks)
            dev_summary_writer.add_summary(summary, current_step)
            test_loss += loss
            test_acc += acc
            pbar.set_postfix(cost=loss, accuracy=acc)
            current_step += 1
        train_loss /= len(train_input_ids) / config_parm.batch_size
        train_acc /= len(train_input_ids) / config_parm.batch_size
        test_loss /= len(test_input_ids) / config_parm.batch_size
        test_acc /= len(test_input_ids) / config_parm.batch_size

        if test_acc > CURRENT_ACC:
            print(
                'epoch: %d, pass acc: %f, current acc: %f'
                % (EPOCH, CURRENT_ACC, test_acc)
            )
            saver.save(sess, checkpoint_dir, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(checkpoint_dir))
            CURRENT_ACC = test_acc
            CURRENT_CHECKPOINT = 0
        else:
            CURRENT_CHECKPOINT += 1

        print('time taken:', time.time() - lasttime)
        print(
            'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
            % (EPOCH, train_loss, train_acc, test_loss, test_acc)
        )
        EPOCH += 1
        if EPOCH == 2:
            break

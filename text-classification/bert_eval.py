import tensorflow as tf
import numpy as np
import data_helpers
from model.bert_fune import Model, BertConfig
from bert import tokenization
from tqdm import tqdm
from sklearn import metrics

config_parm = BertConfig()
config_parm.is_training = False
config_parm.keep_prob = 1.
tokenization.validate_case_matches_checkpoint(True, config_parm.BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(
      vocab_file=config_parm.BERT_VOCAB, do_lower_case=True)

print("Loading data...")

x_text, y, y_onehot = data_helpers.load_data_and_labels(config_parm, is_training=False)

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


config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配

#清除默认图的堆栈，并设置全局图为默认图
tf.reset_default_graph()
with tf.Session(config=config) as sess:
    model = Model(config_parm)
    ckpt = tf.train.get_checkpoint_state("runs/{}".format(config_parm.model_dir))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(config_parm.model_dir))

    all_predictions = []
    pbar = tqdm(
        range(0, len(input_ids), config_parm.batch_size), desc='train minibatch loop'
    )

    for i in pbar:
        index = min(i + config_parm.batch_size, len(input_ids))
        batch_x = input_ids[i: index]
        batch_masks = input_masks[i: index]
        batch_segment = segment_ids[i: index]
        batch_y = y[i: index]
        batch_predictions = model.infer(sess, batch_x, batch_y, batch_segment, batch_masks)
        all_predictions = np.concatenate([all_predictions, batch_predictions])

    print(metrics.classification_report(y, all_predictions, digits=4))



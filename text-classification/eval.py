import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
from sklearn import metrics
from model.bert import Model, BertConfig

# CHANGE THIS: Load data. Load your own data here
config_parm = BertConfig()
if config_parm.test_file:
    x_text, y, y_onehot = data_helpers.load_data_and_labels(config_parm, is_training=False)
    word2id, id2word = data_helpers.loadDataset(config_parm)

    y_test = y
    y_test = list(y_test)
else:
    print("please set a test_file path to test model")
    x_text = ""
    y_test = ""

# Map data into vocabulary

print("\nEvaluating...\n")

# reset graph
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
tf.reset_default_graph()
graph_1 = tf.Graph()
with graph_1.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)

    with tf.Session(config=session_conf) as sess:
        model = Model(
            # size_layer=config_parm.size_layer,
            # num_layers=config_parm.num_layers,
            seq_len=config_parm.seq_len,
            embedding_size=config_parm.embedding_size,
            num_classes=config_parm.num_classes,
            learning_rate=config_parm.learning_rate,
            vocab_size=len(word2id),
            size_layer=config_parm.size_layer,
            num_layers=config_parm.num_layers,

        )
        ckpt = tf.train.get_checkpoint_state("runs/{}".format(config_parm.model_dir))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(config_parm.model_dir))


        # Generate batches for one epoch
        batches = data_helpers.batch_iter(
            list(data_helpers.str_idx(x_text, word2id, maxlen=config_parm.seq_len)),
            config_parm.batch_size,
            num_epochs=1)

        # Collect the predictions here
        all_predictions = []
        scores_sum = []
        for ind, x_test_batch in enumerate(batches):
            batch_predictions = model.infer(sess, x_test_batch)
            all_predictions = np.concatenate([all_predictions, batch_predictions])


        print(metrics.classification_report(y_test, all_predictions, digits=4))



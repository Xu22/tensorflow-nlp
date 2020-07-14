#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
# from model.basic_rnn import Model, RNNConfig
# from model.only_attention import Model, ATTConfig
from model.bert import Model, BertConfig
import pickle
from collections import defaultdict

# import pickle
from tensorflow.contrib import learn
from sklearn import metrics
import jieba
from hanziconv import HanziConv
# from evaluation import *
import code
'  '
# Parameters
# ==================================================
# Data loading params
# config_parm = RNNConfig()
# config_parm = ATTConfig()
config_parm = BertConfig()

# Load data
#数据准备，加载数据
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y, y_onehot = data_helpers.load_data_and_labels(config_parm, is_training=True)

#建立词汇
max_document_length = max([len(x.split(" ")) for x in x_text])#计算最长的词汇长度
# print("最大长度：", max_document_length)
words = " ".join(x_text).split()
vocab_size = len(list(set(" ".join(x_text).split())))
data, count, dictionary, rev_dictionary = data_helpers.build_dataset(words, vocab_size)

GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']

dic = defaultdict()
dic['word2id'] = dictionary
dic['id2word'] = {v: k for k, v in dictionary.items()}

#dumps 写入词汇
data_helpers.uploadDataset(config_parm.vocab_dir, dic)

# x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
#随机清洗数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))#np.arange生成随机序列
x_shuffled = np.array(x_text)[shuffle_indices]
# y_shuffled = y[shuffle_indices]
y_shuffled = np.array(y)[shuffle_indices]


# Split train/test set
# 将数据按训练train和测试dev分块
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(config_parm.dev_sample_percentage * float(len(y)))
print("************************************************************",dev_sample_index)
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(vocab_size))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
#训练开始
# ==================================================
# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(
#       allow_soft_placement=config.allow_soft_placement,
#       log_device_placement=config.log_device_placement)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
with tf.Session(config=config) as sess:
    #卷积池化网络导入
    model = Model(
        # rnn_size=config_parm.rnn_size,
        # num_layers=config_parm.num_layers,
        seq_len=config_parm.seq_len,
        embedding_size=config_parm.embedding_size,
        num_classes=config_parm.num_classes,
        learning_rate=config_parm.learning_rate,
        vocab_size=len(dictionary),
        size_layer = config_parm.size_layer,
        num_layers = config_parm.num_layers,

        )
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", config_parm.model_dir))

    ckpt = tf.train.get_checkpoint_state(out_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # tf.metrics.accuracy会产生两个局部变量
    # 模型保存目录，如果没有则创建
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Summaries for loss and accuracy
    # 损失函数和准确率的参数保存
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

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

    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(data_helpers.str_idx(x_train, dictionary, maxlen=config_parm.seq_len), y_train)),
        config_parm.batch_size,
        config_parm.num_epochs)
    # Training loop. For each batch...

    for batch in batches:
        x_batch, y_batch = zip(*batch)#按batch把数据拿进来
        _, summary = model.train(sess, x_batch, y_batch)
        train_summary_writer.add_summary(summary, current_step)
        current_step += 1
        if current_step % config_parm.evaluate_every == 0: # 每FLAGS.evaluate_every次每100执行一次测试
            print("\nEvaluation:")
            loss, summary = model.eval(sess, data_helpers.str_idx(x_dev, dictionary, maxlen=config_parm.seq_len), y_dev)
            dev_summary_writer.add_summary(summary, current_step)

        if current_step % config_parm.checkpoint_every == 0:# 每checkpoint_every次执行一次保存模型
            # checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_dir, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(checkpoint_dir))


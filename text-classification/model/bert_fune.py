import tensorflow as tf
from sklearn import metrics

import sys
sys.path.append("..")
from bert import modeling

class BertConfig(object):

    data_file = "data/train_data.txt"
    test_file = "data/test_data.txt"
    model_dir = 'bert'
    vocab_dir = 'data/vocab.pickle'
    embedding_size = 64
    seq_len = 120
    size_layer = 128
    num_layers = 2
    num_classes = 20
    learning_rate = 2e-5
    is_training = True
    batch_size = 8
    num_epochs = 2
    evaluate_every = 10
    checkpoint_every = 10
    num_checkpoints = 5
    allow_soft_placement = True
    log_device_placement = True

    BERT_VOCAB = 'chinese_L-12_H-768_A-12/vocab.txt'
    BERT_INIT_CHKPNT = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    BERT_CONFIG = 'chinese_L-12_H-768_A-12/bert_config.json'

    num_filters=128        #number of convolution kernel
    filter_sizes=[2,3,4]   #size of convolution kernel
    hidden_dim=128         #number of fully_connected layer units
    keep_prob=0.5        #droppout
    clip= 5.0              #gradient clipping threshold

class Model(object):
    """
    A RNN for text classification.

    """
    def __init__(
            self, config):

        self.bert_config = modeling.BertConfig.from_json_file(config.BERT_CONFIG)
        self.num_classes = config.num_classes
        self.learning_rate = config.learning_rate
        self.is_training = config.is_training
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.hidden_dim = config.hidden_dim
        self.clip = config.clip
        self.keep_prob = config.keep_prob
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.build_model()
    def build_model(self):
        # Placeholders for input, output
        self.input_ids = tf.placeholder(tf.int32, [None, None])
        self.input_mask = tf.placeholder(tf.int32, [None, None])
        self.segment_ids = tf.placeholder(tf.int32, [None, None])
        self.label_ids = tf.placeholder(tf.int32, [None])

        with tf.name_scope('bert'):
            model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=False)
            embedding_inputs = model.get_sequence_output()

        '''用三个不同的卷积核进行卷积和池化，最后将三个结果concat'''
        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                    conv = tf.layers.conv1d(embedding_inputs, self.num_filters, filter_size, name='conv1d')
                    pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                    pooled_outputs.append(pooled)

            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 1)
            outputs = tf.reshape(h_pool, [-1, num_filters_total])

        '''加全连接层和dropuout层'''
        with tf.name_scope('fc'):
            fc = tf.layers.dense(outputs, self.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        '''logits'''
        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.num_classes, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, output_type=tf.int32)

        '''计算loss，因为输入的样本标签不是one_hot的形式，需要转换下'''
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.label_ids, depth=self.num_classes, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)

        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        '''accuracy'''
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.label_ids, self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("acc", self.acc)
        self.summary_op = tf.summary.merge_all()


    def train(self, sess, x_batch, y_batch, batch_segment, batch_masks):

        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        feed_dict = {
            self.input_ids: x_batch,
            self.label_ids: y_batch,
            self.input_mask: batch_masks,
            self.segment_ids: batch_segment
        }
        _, loss, y_pred_cls, summary, acc = sess.run([self.optim, self.loss, self.y_pred_cls, self.summary_op, self.acc],
                                                      feed_dict=feed_dict)

        print(metrics.classification_report(y_batch, y_pred_cls))
        return acc, loss, summary
    def eval(self, sess, x_batch, y_batch, batch_segment, batch_masks):

        feed_dict = {
            self.input_ids: x_batch,
            self.label_ids: y_batch,
            self.input_mask: batch_masks,
            self.segment_ids: batch_segment
        }
        _, loss, y_pred_cls, summary, acc = sess.run([self.optim, self.loss, self.y_pred_cls, self.summary_op, self.acc],
                                                      feed_dict=feed_dict)

        print(metrics.classification_report(y_batch, y_pred_cls))
        return acc, loss, summary



    def infer(self, sess, x_batch, y_batch, batch_segment, batch_masks):
        feed_dict = {
            self.input_ids: x_batch,
            self.input_mask: batch_masks,
            self.segment_ids: batch_segment
        }
        y_pred_cls = sess.run(self.y_pred_cls, feed_dict=feed_dict)


        return y_pred_cls



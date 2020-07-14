import tensorflow as tf
import numpy as np
from sklearn import metrics

class SiameseConfig(object):
    dev_sample_percentage = 0.05
    data_file = "data/train_data.txt"
    test_file = "data/test_data.txt"
    model_dir = 'siamese'
    vocab_dir = 'data/vocab.pickle'

    embedding_size = 64
    seq_len = 100
    size_layer = 128
    num_layers = 2
    num_classes = 20
    learning_rate = 0.001

    batch_size = 100
    num_epochs = 5
    evaluate_every = 10
    checkpoint_every = 10
    num_checkpoints = 5
    allow_soft_placement = True
    log_device_placement = True



class Model(object):
    """
    A RNN for text classification.

    """
    def __init__(
      self, seq_len, embedding_size,
            vocab_size, num_classes, learning_rate, size_layer, num_layers, margin=1.):

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.size_layer = size_layer
        self.num_layers = num_layers
        self.margin = margin

        self.build_model()

    def cells(self, reuse=False):
        return tf.nn.rnn_cell.BasicRNNCell(self.size_layer, reuse=reuse)

    def rnn(self, embedded, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([self.cells() for _ in range(self.num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells, embedded, dtype=tf.float32)
            W = tf.get_variable('w', shape=(self.size_layer, self.num_classes), initializer=tf.orthogonal_initializer())
            b = tf.get_variable('b', shape=(self.num_classes), initializer=tf.zeros_initializer())
            return tf.matmul(outputs[:, -1], W) + b

    def build_model(self):
        # Placeholders for input, output
        self.INPUT_1 = tf.placeholder(tf.int32, [None, None])
        self.INPUT_2 = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        encoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1, 1))
        input1_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.INPUT_1)
        input2_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.INPUT_2)
        self.logits_1 = self.rnn(input1_embedded, False)
        self.logits_2 = self.rnn(input2_embedded, True)
        self.distance = tf.sqrt(tf.reduce_sum(tf.pow(self.logits_1 - self.logits_2, 2), 1, keep_dims=True))
        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.logits_1), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.logits_2), 1, keep_dims=True))))

        tmp = self.Y * tf.square(self.distance)
        tmp2 = (1 - self.Y) * tf.square(tf.maximum((self.margin - self.distance), 0))
        self.loss = tf.reduce_mean(tmp + tmp2) / 2
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                        name="temp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

        # Define optimizer
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True)#定义优化器
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, x_batch_1, x_batch_2, y_batch):
        '''
        :param sess:
        :param x_batch_1: shape(batch_size, seq_len)
        :param x_batch_2: shape(batch_size, seq_len)
        :param y_batch: shape(batch_size, 1)    同一类别为1，非同一类为0
        :return: distance 可以根据distance 来找寻最相似的两句话
        '''
        feed_dict = {
            self.INPUT_1: x_batch_1,
            self.INPUT_2: x_batch_2,
            self.Y: y_batch,

        }
        _, loss, accuracy, summary = sess.run([self.train_op, self.loss, self.accuracy, self.summary_op],
                                                      feed_dict=feed_dict)
        # y_label = np.argmax(y_batch, 1)
        # prediction = np.argmax(logits, 1)
        # print(metrics.classification_report(y_batch, prediction))
        return loss, summary
    def eval(self, sess, x_batch_1, x_batch_2, y_batch):
        '''
        :param sess:
        :param x_batch_1: shape(batch_size, seq_len)
        :param x_batch_2: shape(batch_size, seq_len)
        :param y_batch: shape(batch_size, 1)    同一类别为1，非同一类为0
        :return: distance 可以根据distance 来找寻最相似的两句话
        '''
        feed_dict = {
            self.INPUT_1: x_batch_1,
            self.INPUT_2: x_batch_2,
            self.Y: y_batch,

        }
        _, loss, accuracy, summary = sess.run([self.train_op, self.loss, self.accuracy, self.summary_op],
                                                      feed_dict=feed_dict)
        # y_label = np.argmax(y_batch, 1)
        # prediction = np.argmax(logits, 1)
        # print(metrics.classification_report(y_batch, prediction))
        return loss, summary

    def infer(self, sess, x_batch_1, x_batch_2):
        '''
        :param sess:
        :param x_batch_1: shape(batch_size, seq_len)
        :param x_batch_2: shape(batch_size, seq_len)
        :param y_batch: shape(batch_size, 1)    同一类别为1，非同一类为0
        :return: distance 可以根据distance 来找寻最相似的两句话
        '''
        feed_dict = {
            self.INPUT_1: x_batch_1,
            self.INPUT_2: x_batch_2,
        }
        _, distance, summary = sess.run([self.train_op, self.distance, self.summary_op],
                                                      feed_dict=feed_dict)
        # y_label = np.argmax(y_batch, 1)
        # prediction = np.argmax(logits, 1)
        # print(metrics.classification_report(y_batch, prediction))
        return distance, summary

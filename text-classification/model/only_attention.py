import tensorflow as tf
import numpy as np
from sklearn import metrics

class ATTConfig(object):
    dev_sample_percentage = 0.1
    data_file = "data/train_data.txt"
    model_dir = 'only_att'
    embedding_size = 64
    seq_len = 50
    size_layer = 128
    num_classes = 20
    learning_rate = 0.001

    batch_size = 100
    num_epochs = 5
    evaluate_every = 10
    checkpoint_every = 10
    num_checkpoints = 5
    allow_soft_placement = True
    log_device_placement = True

def sinusoidal_positional_encoding(inputs, maxlen, zero_pad=False, scale=False):
    E = inputs.get_shape().as_list()[-1] # static, (embedding_size)
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] #dynamic  (batch_size, seq_len)
    position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  #(N, T)
    position_enc = np.array([[pos / np.power(10000, 2.*i/E) for i in range(E)] for pos in range(maxlen)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    lookup_table = tf.convert_to_tensor(position_enc, tf.float32) #(maxlen, E)
    if zero_pad:
        lookup_table = tf.concat([tf.zeros([1, E]), lookup_table[1:, :]], axis=0)
    outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
    if scale:
        outputs = outputs * E ** 0.5
    return outputs


class Model(object):
    """
    A RNN for text classification.

    """
    def __init__(
      self, seq_len, embedding_size,
            vocab_size, num_classes, learning_rate):

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.build_model()
    def build_model(self):
        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x") # input_x输入语料,待训练的内容,维度是sequence_length,"N个词构成的N维向量"
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y") # input_y输入语料,待训练的内容标签,维度是num_classes,"正面 || 负面"


        # Embedding layer
        # 指定运算结构的运行位置在cpu非gpu,因为"embedding"无法运行在gpu
        # 通过tf.name_scope指定"embedding"
        #在某个tf.name_scope()指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域
        #即这里面W获取时，应为“embedding/W”
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)   #shape  [batch_size, seq_len, embedding_size]

            x = embedded_chars
            x += sinusoidal_positional_encoding(x, self.seq_len)

        masks = tf.sign(embedded_chars[:, :, 0])
        align = tf.squeeze(tf.layers.dense(x, 1, tf.tanh), -1)
        paddings = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.expand_dims(tf.nn.softmax(align), -1)
        x = tf.squeeze(tf.matmul(tf.transpose(x, [0, 2, 1]), align), -1)

        #输出层
        with tf.variable_scope("output"):

            self.logits = tf.layers.dense(x, self.num_classes, name="scores")#得分函数
            self.predictions = tf.argmax(self.logits, 1, name="predictions")#预测结果

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

        # Define optimizer
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)#定义优化器
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # =================================保存模型
        self.saver = tf.train.Saver(tf.global_variables())
    def train(self, sess, x_batch, y_batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch
        }
        _, loss, summary, prediction = sess.run([self.train_op, self.loss, self.summary_op, self.predictions], feed_dict=feed_dict)
        y_label = np.argmax(y_batch, 1)
        print(metrics.classification_report(y_label, prediction))

    def eval(self, sess, x_batch, y_batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch
                     }
        _, loss, summary, prediction = sess.run([self.train_op, self.loss, self.summary_op, self.predictions], feed_dict=feed_dict)
        y_label = np.argmax(y_batch, 1)
        print(metrics.classification_report(y_label, prediction))

        return loss, summary

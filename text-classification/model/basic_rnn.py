import tensorflow as tf
import numpy as np
from sklearn import metrics

class RNNConfig(object):
    dev_sample_percentage = 0.1
    data_file = "data/train_data.txt"
    model_dir = 'basic_rnn'
    batch_size = 100
    num_epochs = 5
    evaluate_every = 10
    checkpoint_every = 10
    num_checkpoints = 5
    allow_soft_placement = True
    log_device_placement = True

    seq_len = 50
    embedding_size = 64
    num_layers = 1
    size_layer = 128
    num_classes = 20
    learning_rate = 0.001



#定义rnn网络实现的类

class Model(object):
    """
    A RNN for text classification.

    """
    def __init__(
      self, rnn_size, num_layers, embedding_size,
            vocab_size, num_classes, learning_rate, keep_prob_placeholder):

        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # 执行模型构建部分的代码
        self.build_model()

    def _create_rnn_cell(self, reuse=False):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            '''
            tf.nn.rnn_cell.BasicRNNCell
            '''
            if self.model_flag == "rnn":
                single_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size, reuse=reuse)
            elif self.model_flag == "lstm":
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
            elif self.model_flag == "gru":
                single_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)


            return single_cell

        #列表中每个元素都是调用single_rnn_cell函数
        cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        # 添加dropout
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob_placeholder)
        return cell
    def _create_biRnn_cell(self, reuse=False):
        def single_rnn_cell(size):
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            return single_cell
        for n in range(self.num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=single_rnn_cell(self.rnn_size),
                cell_bw=single_rnn_cell(self.rnn_size),
                inputs=self.embedded_chars,
                dtype=tf.float32,
                scope='bidirectional_rnn_%d'%(n))
            self.embedded_chars = tf.concat((out_fw, out_bw), 2)
        return self.embedded_chars
    def BiRNN(self):

        # Prepare data shape to match `static_rnn` function requirements
        '''
        x [batch_size, seq_length, embedding_size]
        '''
        self.embedded_chars = tf.unstack(tf.transpose(self.embedded_chars, perm=[1, 0, 2]))
        '''
        x [seq_length, batch_size, embedding_size]
        '''

        lstm_fw_cell_m = self._create_rnn_cell()
        lstm_bw_cell_m = self._create_rnn_cell()
        with tf.name_scope("bw"), tf.variable_scope("bw"):
            '''
            tf.nn.static_bidirectional_rnn
            return (outputs, output_state_fw, output_state_bw)
            '''
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.embedded_chars, dtype=tf.float32)
        return outputs[-1]

    def build_model(self):
        print('building model... ...')
        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32, [None, None],
                                      name="input_x")  # input_x输入语料,待训练的内容,维度是sequence_length,"N个词构成的N维向量"
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes],
                                      name="input_y")  # input_y输入语料,待训练的内容标签,维度是num_classes,"正面 || 负面"
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
        self.model_flag = "rnn"
        self.single_rnn = True
        # model_type = ["rnn", "lstm", "gru"]
        # Embedding layer
        # 指定运算结构的运行位置在cpu非gpu,因为"embedding"无法运行在gpu
        # 通过tf.name_scope指定"embedding"
        #在某个tf.name_scope()指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域
        #即这里面W获取时，应为“embedding/W”
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  #batch_size, seq_len, embedding_size


        with tf.name_scope("rnn"):
            if self.single_rnn:
                rnn_cells = self._create_rnn_cell()
                self.rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cells, self.embedded_chars, dtype=tf.float32)
            else:
                self.rnn_outputs = self._create_biRnn_cell()
            rnn_outputs = self.rnn_outputs[:, -1]


        #输出层
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.rnn_size, self.num_classes],#前面连扁平化后的池化操作
                initializer=tf.contrib.layers.xavier_initializer())# 定义初始化方式
            b = tf.get_variable('b', shape=([self.num_classes]))

            self.logits = tf.nn.xw_plus_b(rnn_outputs, W, b, name="scores")#得分函数
            self.predictions = tf.argmax(self.logits, 1, name="predictions")#预测结果

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            l2 = sum(1e-5 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)) + l2

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
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.keep_prob_placeholder: 0.5
                     }
        _, loss, summary, prediction = sess.run([self.train_op, self.loss, self.summary_op, self.predictions], feed_dict=feed_dict)
        y_label = np.argmax(y_batch, 1)
        print(metrics.classification_report(y_label, prediction))

        return loss, summary
    def eval(self, sess, x_batch, y_batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.keep_prob_placeholder: 1.
                     }
        _, loss, summary, prediction = sess.run([self.train_op, self.loss, self.summary_op, self.predictions], feed_dict=feed_dict)
        y_label = np.argmax(y_batch, 1)
        print(metrics.classification_report(y_label, prediction))

        return loss, summary

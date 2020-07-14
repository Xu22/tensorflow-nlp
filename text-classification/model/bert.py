import tensorflow as tf
import numpy as np
from sklearn import metrics
# from . import bert_model as modeling
from bert import run_classifier, optimization, tokenization, modeling

class BertConfig(object):
    dev_sample_percentage = 0.05
    data_file = "data/train_data.txt"
    test_file = "data/test_data.txt"
    model_dir = 'bert'
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

    BERT_VOCAB = '../chinese_L-12_H-768_A-12/vocab.txt'
    BERT_INIT_CHKPNT = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
    BERT_CONFIG = '../chinese_L-12_H-768_A-12/bert_config.json'

def create_model(
    bert_config,
    is_training,
    input_ids,
    input_mask,
    segment_ids,
    labels,
    num_labels,
    use_one_hot_embeddings,
    reuse_flag = False,
):
    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = input_mask,
        token_type_ids = segment_ids,
        use_one_hot_embeddings = use_one_hot_embeddings,
    )

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    with tf.variable_scope('weights', reuse = reuse_flag):
        output_weights = tf.get_variable(
            'output_weights',
            [num_labels, hidden_size],
            initializer = tf.truncated_normal_initializer(stddev = 0.02),
        )
        output_bias = tf.get_variable(
            'output_bias', [num_labels], initializer = tf.zeros_initializer()
        )

    with tf.variable_scope('loss'):
        def apply_dropout_last_layer(output_layer):
            output_layer = tf.nn.dropout(output_layer, keep_prob = 0.9)
            return output_layer

        def not_apply_dropout(output_layer):
            return output_layer

        output_layer = tf.cond(
            is_training,
            lambda: apply_dropout_last_layer(output_layer),
            lambda: not_apply_dropout(output_layer),
        )
        logits = tf.matmul(output_layer, output_weights, transpose_b = True)
        print(
            'output_layer:',
            output_layer.shape,
            ', output_weights:',
            output_weights.shape,
            ', logits:',
            logits.shape,
        )

        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = labels, logits = logits
        )
        loss = tf.reduce_mean(loss)
        correct_pred = tf.equal(tf.argmax(logits, 1, output_type = tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return loss, logits, probabilities, model, accuracy

class Model(object):
    """
    A RNN for text classification.

    """
    def __init__(
      self, seq_len, embedding_size,
            vocab_size, num_classes, learning_rate, size_layer, num_layers):

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.size_layer = size_layer
        self.num_layers = num_layers

        self.build_model()
    def build_model(self):
        # Placeholders for input, output

        BERT_VOCAB = '../chinese_L-12_H-768_A-12/vocab.txt'
        BERT_INIT_CHKPNT = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
        BERT_CONFIG = '../chinese_L-12_H-768_A-12/bert_config.json'
        tokenization.validate_case_matches_checkpoint(True, '')
        bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=BERT_VOCAB, do_lower_case=True)

        bert_config = modeling.BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.size_layer,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.size_layer // 4,
            intermediate_size=self.size_layer * 2,
        )

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_len])
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_len])
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_len])
        self.label_ids = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)

        use_one_hot_embeddings = False
        self.loss, self.logits, probabilities, model, self.accuracy = create_model(
            bert_config,
            self.is_training,
            self.input_ids,
            self.input_mask,
            self.segment_ids,
            self.label_ids,
            self.num_classes,
            use_one_hot_embeddings,
        )
        global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.optimizer = tf.contrib.layers.optimize_loss(
            self.loss,
            global_step=global_step,
            learning_rate=self.learning_rate,
            optimizer='Adam',
            clip_gradients=3.0,
        )
        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, x_batch, y_batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        np_mask = np.ones((len(x_batch), self.seq_len), dtype = np.int32)
        np_segment = np.ones((len(x_batch), self.seq_len), dtype = np.int32)
        feed_dict = {
            self.input_ids: x_batch,
            self.label_ids: y_batch,
            self.input_mask: np_mask,
            self.segment_ids: np_segment,
            self.is_training: True
        }
        _, loss, accuracy, logits, summary = sess.run([self.optimizer, self.loss, self.accuracy, self.logits, self.summary_op],
                                                      feed_dict=feed_dict)
        # y_label = np.argmax(y_batch, 1)
        prediction = np.argmax(logits, 1)
        print(metrics.classification_report(y_batch, prediction))
        return loss, summary
    def eval(self, sess, x_batch, y_batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        np_mask = np.ones((len(x_batch), self.seq_len), dtype = np.int32)
        np_segment = np.ones((len(x_batch), self.seq_len), dtype = np.int32)
        feed_dict = {
            self.input_ids: x_batch,
            self.label_ids: y_batch,
            self.input_mask: np_mask,
            self.segment_ids: np_segment,
            self.is_training: False
        }
        _, loss, accuracy, logits, summary = sess.run([self.optimizer, self.loss, self.accuracy, self.logits, self.summary_op],
                                                      feed_dict=feed_dict)
        # y_label = np.argmax(y_batch, 1)
        prediction = np.argmax(logits, 1)
        print(metrics.classification_report(y_batch, prediction))
        return loss, summary

    def infer(self, sess, x_batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        np_mask = np.ones((len(x_batch), self.seq_len), dtype=np.int32)
        np_segment = np.ones((len(x_batch), self.seq_len), dtype=np.int32)
        feed_dict = {
            self.input_ids: x_batch,
            self.input_mask: np_mask,
            self.segment_ids: np_segment,
            self.is_training: False
        }
        logits = sess.run(self.logits, feed_dict=feed_dict)
        # y_label = np.argmax(y_batch, 1)
        prediction = np.argmax(logits, 1)
        # print(metrics.classification_report(y_batch, prediction))
        return prediction

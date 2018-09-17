import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, OutputProjectionWrapper, DropoutWrapper

class RNNModel(object,):

    def __init__(self, sequence_length, RNN_size):

        # define placeholder
        self.input_x = tf.placeholder(tf.float32, shape=[None, sequence_length], name='Input_x')
        self.output_y = tf.placeholder(tf.float32, shape=[None, 1], name='output_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # expand inputs
        self.input_sequence = tf.expand_dims(self.input_x, axis=-1)

        # RNN part
        with tf.name_scope("RNN"):
            cell = BasicRNNCell(RNN_size)
            cell = DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.outputs, self.states = tf.nn.dynamic_rnn(cell, self.input_sequence, dtype=tf.float32)

        # output and loss
        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([RNN_size, 1], mean=2, stddev=0.1), name='output_weight')
            b = tf.Variable(tf.truncated_normal([1], mean=2, stddev=0.1), name='bias')

            self.output = tf.nn.xw_plus_b(self.states, W, b)
            self.loss = tf.reduce_mean(tf.square(self.output_y - self.output))
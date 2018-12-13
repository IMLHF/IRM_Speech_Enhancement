# speech enhancement

import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn.python.ops import rnn
from FLAGS import NNET_PARAM
from utils import tf_tool


class SE_MODEL(object):
  def __init__(self,
               inputs_batch,
               label_batch,
               lengths_batch,
               theta_x_batch=None,
               theta_y_batch=None,
               infer=False):
    self._inputs = inputs_batch
    self._mixed = self._inputs
    self._labels = label_batch
    self._lengths = lengths_batch

    self.batch_size = tf.shape(self._lengths)[0]
    self._model_type = NNET_PARAM.MODEL_TYPE

    outputs = self._inputs

    def lstm_cell():
      return tf.contrib.rnn.LSTMCell(
          NNET_PARAM.RNN_SIZE, forget_bias=1.0, use_peepholes=True,
          initializer=tf.contrib.layers.xavier_initializer(),
          state_is_tuple=True, activation=NNET_PARAM.LSTM_ACTIVATION)
    lstm_attn_cell = lstm_cell
    if not infer and NNET_PARAM.KEEP_PROB < 1.0:
      def lstm_attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=NNET_PARAM.KEEP_PROB)

    def GRU_cell():
      return tf.contrib.rnn.GRUCell(
          NNET_PARAM.RNN_SIZE,
          # kernel_initializer=tf.contrib.layers.xavier_initializer(),
          activation=NNET_PARAM.LSTM_ACTIVATION)
    GRU_attn_cell = lstm_cell
    if not infer and NNET_PARAM.KEEP_PROB < 1.0:
      def GRU_attn_cell():
        return tf.contrib.rnn.DropoutWrapper(GRU_cell(), output_keep_prob=NNET_PARAM.KEEP_PROB)

    if NNET_PARAM.MODEL_TYPE.upper() == 'BLSTM':
      with tf.variable_scope('BLSTM'):

        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_attn_cell() for _ in range(NNET_PARAM.RNN_LAYER)], state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_attn_cell() for _ in range(NNET_PARAM.RNN_LAYER)], state_is_tuple=True)

        lstm_fw_cell = lstm_fw_cell._cells
        lstm_bw_cell = lstm_bw_cell._cells
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=lstm_fw_cell,
            cells_bw=lstm_bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result
    if NNET_PARAM.MODEL_TYPE.upper() == 'BGRU':
      with tf.variable_scope('BGRU'):

        gru_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [GRU_attn_cell() for _ in range(NNET_PARAM.RNN_LAYER)], state_is_tuple=True)
        gru_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [GRU_attn_cell() for _ in range(NNET_PARAM.RNN_LAYER)], state_is_tuple=True)

        gru_fw_cell = gru_fw_cell._cells
        gru_bw_cell = gru_bw_cell._cells
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=gru_fw_cell,
            cells_bw=gru_bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result

    with tf.variable_scope('fullconnectOut'):
      if self._model_type.upper()[0] == 'B':  # bidirection
        outputs = tf.reshape(outputs, [-1, 2*NNET_PARAM.RNN_SIZE])
        in_size = 2*NNET_PARAM.RNN_SIZE
      out_size = NNET_PARAM.OUT_SIZE
      weights = tf.get_variable('weights1', [in_size, out_size],
                                initializer=tf.random_normal_initializer(stddev=0.01))
      biases = tf.get_variable('biases1', [out_size],
                               initializer=tf.constant_initializer(0.0))
      irm = tf.nn.relu(tf.matmul(outputs, weights) + biases)
      self._activations = tf.reshape(
          irm, [self.batch_size, -1, NNET_PARAM.OUTPUT_SIZE])

      self._cleaned = self._activations*self._mixed

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if infer:
      return

    if NNET_PARAM.MASK_TYPE == 'PSIRM':
      self._labels *= tf.cos(theta_x_batch-theta_y_batch)

    self._loss = tf.losses.mean_squared_error(self._labels,self._cleaned)

    if tf.get_variable_scope().reuse:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      NNET_PARAM.CLIP_NORM)
    optimizer = tf.train.AdamOptimizer(self.lr)
    #optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name='new_learning_rate')
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def cleaned(self):
    return self._cleaned

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def lengths(self):
    return self._lengths

  @property
  def lr(self):
    return self._lr

  @property
  def loss(self):
    return self._loss

  @property
  def train_op(self):
    return self._train_op

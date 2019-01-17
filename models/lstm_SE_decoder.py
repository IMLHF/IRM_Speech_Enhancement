# speech enhancement

import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn.python.ops import rnn
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM as DATA_PARAM
from utils import tf_tool
from dataManager import mixed_aishell_tfrecord_io as data_tool


class SE_MODEL_decoder(object):
  def __init__(self,
               inputs_batch,
               lengths_batch,
               infer=False):
    self._inputs = inputs_batch
    self._mixed = self._inputs
    self._lengths = lengths_batch

    self.batch_size = tf.shape(self._lengths)[0]
    self._model_type = NNET_PARAM.MODEL_TYPE

    outputs = self._inputs

    def lstm_cell():
      return tf.contrib.rnn.LSTMCell(
          NNET_PARAM.RNN_SIZE, forget_bias=1.0, use_peepholes=True,
          num_proj=NNET_PARAM.LSTM_num_proj,
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
        outputs = tf.reshape(outputs, [-1, 2*NNET_PARAM.LSTM_num_proj])
        in_size = 2*NNET_PARAM.LSTM_num_proj
      out_size = NNET_PARAM.OUTPUT_SIZE
      weights = tf.get_variable('weights1', [in_size, out_size],
                                initializer=tf.random_normal_initializer(stddev=0.01))
      biases = tf.get_variable('biases1', [out_size],
                               initializer=tf.constant_initializer(0.0))
      mask = tf.nn.relu(tf.matmul(outputs, weights) + biases)
      self._activations_t = tf.reshape(
          mask, [self.batch_size, -1, NNET_PARAM.OUTPUT_SIZE])

      # mask clip
      self._activations = self._activations_t
      # self._activations = tf.clip_by_value(self._activations_t,-1,1.5)

      masked_mag = None
      if DATA_PARAM.FEATURE_TYPE == 'LOG_MAG' and DATA_PARAM.MASK_ON_MAG_EVEN_LOGMAG:
        mag = data_tool.rmNormalization(self._mixed, eager=False)

        # norm to (0,1), 大数乘小数会有误差，mask比较小，所以将mag变小。
        mag = tf.clip_by_value(mag, DATA_PARAM.MAG_NORM_MIN, DATA_PARAM.MAG_NORM_MAX)
        mag -= DATA_PARAM.MAG_NORM_MIN
        mag /= (DATA_PARAM.MAG_NORM_MAX - DATA_PARAM.MAG_NORM_MIN)

        # add mask on magnitude spectrum
        masked_mag = self._activations*mag

        # rm mag norm
        masked_mag = masked_mag*(DATA_PARAM.MAG_NORM_MAX -
                                 DATA_PARAM.MAG_NORM_MIN)+DATA_PARAM.MAG_NORM_MIN

        # change to log_mag feature
        log_masked_mag = tf.log(masked_mag+DATA_PARAM.LOG_BIAS)/tf.log(10.0)
        log_masked_mag = tf.clip_by_value(log_masked_mag,
                                          DATA_PARAM.LOG_NORM_MIN,
                                          DATA_PARAM.LOG_NORM_MAX)
        log_masked_mag -= DATA_PARAM.LOG_NORM_MIN
        log_masked_mag /= (DATA_PARAM.LOG_NORM_MAX - DATA_PARAM.LOG_NORM_MIN)
        self._cleaned = log_masked_mag
      else:
        self._cleaned = self._activations*self._mixed

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if infer:
      if DATA_PARAM.FEATURE_TYPE == 'LOG_MAG' and DATA_PARAM.MASK_ON_MAG_EVEN_LOGMAG:
        self._cleaned = masked_mag
      return


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def cleaned(self):
    return self._cleaned

  @property
  def inputs(self):
    return self._inputs


  @property
  def mask(self):
    return self._activations

  @property
  def lengths(self):
    return self._lengths



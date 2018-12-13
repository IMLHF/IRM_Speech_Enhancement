from pyasv.basic import blocks, layers, model
import tensorflow as tf
from pyasv.config import Config


class IRM(model.Model):
    def __init__(self, config):
        super().__init__(config)

    def inference(self, inp):
        n_hidden = 513
        frames_per_sample = 100
        batch_size = 64
        p_keep_ff = 1
        NEFF = 513
        p_keep_rc = 1
        with tf.variable_scope('BLSTM1'):
            lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, layer_norm=False,
                dropout_keep_prob=p_keep_rc)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_fw_cell, input_keep_prob=1,
                output_keep_prob=p_keep_ff)
            lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, layer_norm=False,
                dropout_keep_prob=p_keep_rc)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_bw_cell, input_keep_prob=1,
                output_keep_prob=p_keep_ff)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inp,
                sequence_length=[frames_per_sample] * batch_size,
                dtype=tf.float32)
            state_concate = tf.concat(outputs,2)

        with tf.variable_scope('BLSTM2'):
            lstm_fw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, layer_norm=False,
                dropout_keep_prob=p_keep_rc)
            lstm_fw_cell2 = tf.contrib.rnn.DropoutWrapper(
                lstm_fw_cell2, input_keep_prob=1,
                output_keep_prob=p_keep_ff)
            lstm_bw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, layer_norm=False,
                dropout_keep_prob=p_keep_rc)
            lstm_bw_cell2 = tf.contrib.rnn.DropoutWrapper(
                lstm_bw_cell2, input_keep_prob=1,
                output_keep_prob=p_keep_ff)
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell2, lstm_bw_cell2, state_concate,
                sequence_length=[frames_per_sample] * batch_size,
                dtype=tf.float32)
            state_concate2 = tf.concat(outputs2, 2)

        with tf.variable_scope('BLSTM3'):
            lstm_fw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, layer_norm=False,
                dropout_keep_prob=p_keep_rc)
            lstm_fw_cell3 = tf.contrib.rnn.DropoutWrapper(
                lstm_fw_cell3, input_keep_prob=1,
                output_keep_prob=p_keep_ff)
            lstm_bw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, layer_norm=False,
                dropout_keep_prob=p_keep_rc)
            lstm_bw_cell3 = tf.contrib.rnn.DropoutWrapper(
                lstm_bw_cell3, input_keep_prob=1,
                output_keep_prob=p_keep_ff)
            outputs3, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell3, lstm_bw_cell3, state_concate2,
                sequence_length=[frames_per_sample] * batch_size,
                dtype=tf.float32)

        with tf.variable_scope('FullConnect') as scope:
            state_concate3 = tf.concat(outputs3,2)
            print(state_concate3.get_shape().as_list())
            out_concate = tf.reshape(state_concate3, [-1, n_hidden * 2])

            print(out_concate.get_shape().as_list())
            out_irm = layers.full_connect(out_concate, name='output', units=NEFF, activation='None')
            print(out_irm.get_shape().as_list())

        return tf.reshape(out_irm, shape=[64, frames_per_sample, NEFF])

    def loss(self, irm, y):
        return tf.losses.mean_squared_error(irm, y)


if __name__ == '__main__':
    inp = tf.placeholder(dtype=tf.float32, shape=[64, 100, 513], name='Input')
    y = tf.placeholder(dtype=tf.float32, shape=[64, 100, 513], name='RateMask')
    con = Config("../config.json")
    with tf.Session() as sess:
        model = IRM(con)
        out = model.inference(inp)
        loss = model.loss(out, y)
        summary = tf.summary.merge_all()
        opt = tf.train.AdamOptimizer(0.0001).minimize(loss)
        writer = tf.summary.FileWriter("./", sess.graph)


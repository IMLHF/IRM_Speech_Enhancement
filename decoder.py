import time
import tensorflow as tf
import numpy as np
import sys
import utils
import utils.audio_tool
import os
import shutil
from models.lstm_SE_decoder import SE_MODEL_decoder
import wave
import gc
from dataManager import mixed_aishell_tfrecord_io as wav_tool
from dataManager.mixed_aishell_tfrecord_io import rmNormalization
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM
from dataManager.mixed_aishell_tfrecord_io import generate_tfrecord, get_batch_use_tfdata


def build_session(ckpt_dir='nnet'):
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch = tf.placeholder(tf.float32,shape=[1,None,NNET_PARAM.INPUT_SIZE],name='x_batch')
        lengths_batch = tf.placeholder(tf.int32,shape=[1],name='lengths_batch')
    with tf.name_scope('model'):
      model = SE_MODEL_decoder(x_batch, lengths_batch,
                               infer=True)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(
        os.path.join(NNET_PARAM.SAVE_DIR, ckpt_dir))
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)
  g.finalize()
  return sess,model


def decode_one_wav(sess, model, wavedata):
  x_spec_t = wav_tool._extract_norm_log_mag_spec(wavedata)
  length = np.shape(x_spec_t)[0]
  x_spec = np.array([x_spec_t], dtype=np.float32)
  lengths = np.array([length], dtype=np.int32)
  cleaned, mask = sess.run(
      [model.cleaned, model.mask],
      feed_dict={
        model.inputs: x_spec,
        model.lengths: lengths,
      })

  if MIXED_AISHELL_PARAM.FEATURE_TYPE == 'LOG_MAG' and MIXED_AISHELL_PARAM.MASK_ON_MAG_EVEN_LOGMAG:
    cleaned = np.array(cleaned)
  else:
    cleaned = np.array(rmNormalization(cleaned))

  cleaned_spec = utils.spectrum_tool.griffin_lim(cleaned, wavedata,
                                                 MIXED_AISHELL_PARAM.NFFT,
                                                 MIXED_AISHELL_PARAM.OVERLAP,
                                                 NNET_PARAM.GRIFFIN_ITERNUM)

  # write restore wave
  reY = utils.spectrum_tool.librosa_istft(
      cleaned_spec, MIXED_AISHELL_PARAM.NFFT, MIXED_AISHELL_PARAM.OVERLAP)
  if NNET_PARAM.decode_output_speaker_volume_amp:  # norm resotred wave
    reY = reY/np.max(np.abs(reY))*32767

  return np.array(reY), mask

if __name__=='__main__':
  f1 = wave.open('jjyykk.wav', 'rb')
  waveData = np.fromstring(f1.readframes(f1.getnframes()),
                           dtype=np.int16)
  f1.close()
  waveData=waveData*1.0

  sess, model = build_session(ckpt_dir="nnet_C11_bias50")
  reY,_ = decode_one_wav(sess,model,waveData)#
  framerate = MIXED_AISHELL_PARAM.FS
  bits = 16
  utils.audio_tool.write_audio('jjyykk_en.wav',
                               reY,
                               framerate,
                               bits, 'wav')

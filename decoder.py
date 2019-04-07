import time
import tensorflow as tf
import numpy as np
import sys
from utils import audio_tool, spectrum_tool, tf_tool
import os
import shutil
from utils.assess import core as pesqexe
from pystoi import stoi
from models.lstm_SE_decoder import SE_MODEL_decoder
import gc
from dataManager import mixed_aishell_tfrecord_io as wav_tool
from dataManager.mixed_aishell_tfrecord_io import rmNormalization
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM
from dataManager.mixed_aishell_tfrecord_io import generate_tfrecord, get_batch_use_tfdata


def build_session(ckpt_dir='nnet',batch_size=1,finalizeG=True):
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch = tf.placeholder(tf.float32,shape=[batch_size,None,NNET_PARAM.INPUT_SIZE],name='x_batch')
        lengths_batch = tf.placeholder(tf.int32,shape=[batch_size],name='lengths_batch')
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
  if finalizeG:
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

  y_mag_estimation = np.array(cleaned[0])
  mask = np.array(mask[0])
  if MIXED_AISHELL_PARAM.FEATURE_TYPE == 'LOG_MAG' and MIXED_AISHELL_PARAM.MASK_ON_MAG_EVEN_LOGMAG:
    y_mag_estimation = np.array(y_mag_estimation)
  else:
    y_mag_estimation = np.array(rmNormalization(y_mag_estimation))

  cleaned_spec = spectrum_tool.griffin_lim(y_mag_estimation, wavedata,
                                           MIXED_AISHELL_PARAM.NFFT,
                                           MIXED_AISHELL_PARAM.OVERLAP,
                                           NNET_PARAM.GRIFFIN_ITERNUM)

  # write restore wave
  reY = spectrum_tool.librosa_istft(
      cleaned_spec, MIXED_AISHELL_PARAM.NFFT, MIXED_AISHELL_PARAM.OVERLAP)
  if NNET_PARAM.decode_output_speaker_volume_amp:  # norm resotred wave
    reY = reY/np.max(np.abs(reY))*32767

  return np.array(reY), mask


def decode_and_getMeature(mixed_file_list, ref_list, sess, model, decode_ans_file, save_audio, ans_file):
  '''
  (mixed_dir,ref_dir,sess,model,'decode_nnet_C001_8_2',False,'xxxans.txt')
  '''
  if os.path.exists(os.path.join(decode_ans_file,ans_file)):
    os.remove(os.path.join(decode_ans_file,ans_file))
  pesq_raw_sum = 0
  pesq_en_sum = 0
  stoi_raw_sum = 0
  stoi_en_sum = 0
  sdr_raw_sum = 0
  sdr_en_sum = 0
  for i, mixed_dir in enumerate(mixed_file_list):
    print('\n',i+1,mixed_dir)
    waveData, sr = audio_tool.read_audio(mixed_dir)
    reY, mask = decode_one_wav(sess,model,waveData)
    abs_max = (2 ** (MIXED_AISHELL_PARAM.AUDIO_BITS - 1) - 1)
    reY = np.where(reY > abs_max, abs_max, reY)
    reY = np.where(reY < -abs_max, -abs_max, reY)
    file_name = mixed_dir[mixed_dir.rfind('/')+1:mixed_dir.rfind('.')]
    if save_audio:
      audio_tool.write_audio(os.path.join(decode_ans_file,
                                          (ckpt+'_%03d_' % (i+1))+mixed_dir[mixed_dir.rfind('/')+1:]),
                             reY,
                             sr)
      spectrum_tool.picture_spec(mask,
                                 os.path.join(decode_ans_file,
                                              (ckpt+'_%03d_' % (i+1))+file_name))

    if i<len(ref_list):
      ref, sr = audio_tool.read_audio(ref_list[i])
      print(' refer: ',ref_list[i])
      len_small = min(len(ref),len(waveData),len(reY))
      ref = np.array(ref[:len_small])
      waveData = np.array(waveData[:len_small])
      reY = np.array(reY[:len_small])
      # sdr
      sdr_raw = audio_tool.cal_SDR(np.array([ref]),
                                   np.array([waveData]))
      sdr_en = audio_tool.cal_SDR(np.array([ref]),
                                  np.array(reY))
      sdr_raw_sum += sdr_raw
      sdr_en_sum += sdr_en
      # pesq
      # pesq_raw = pesq(ref,waveData,sr)
      # pesq_en = pesq(ref,reY,sr)
      pesq_raw = pesqexe.calc_pesq(ref,waveData,sr)
      pesq_en = pesqexe.calc_pesq(ref,reY,sr)
      pesq_raw_sum += pesq_raw
      pesq_en_sum += pesq_en
      # stoi
      stoi_raw = stoi.stoi(ref,waveData,sr)
      stoi_en = stoi.stoi(ref,reY,sr)
      stoi_raw_sum += stoi_raw
      stoi_en_sum += stoi_en
      print("SR = %d" % sr)
      print("PESQ_raw: %.3f, PESQ_en: %.3f, PESQimp: %.3f. " % (pesq_raw,pesq_en,pesq_en-pesq_raw))
      print("SDR_raw: %.3f, SDR_en: %.3f, SDRimp: %.3f. " % (sdr_raw,sdr_en,sdr_en-sdr_raw))
      print("STOI_raw: %.3f, STOI_en: %.3f, STOIimp: %.3f. " % (stoi_raw,stoi_en,stoi_en-stoi_raw))
      sys.stdout.flush()
      with open(os.path.join(decode_ans_file,ans_file),'a+') as f:
        f.write(file_name+'\r\n')
        f.write("    |-PESQ_raw: %.3f, PESQ_en: %.3f, PESQimp: %.3f. \r\n" % (pesq_raw,pesq_en,pesq_en-pesq_raw))
        f.write("    |-SDR_raw: %.3f, SDR_en: %.3f, SDRimp: %.3f. \r\n" % (sdr_raw,sdr_en,sdr_en-sdr_raw))
        f.write("    |-STOI_raw: %.3f, STOI_en: %.3f, STOIimp: %.3f. \r\n" % (stoi_raw,stoi_en,stoi_en-stoi_raw))

  len_list = len(ref_list)
  with open(os.path.join(decode_ans_file,ans_file),'a+') as f:
    f.write('PESQ_raw:%.3f, PESQ_en:%.3f, PESQi_avg:%.3f. \r\n' % (pesq_raw_sum/len_list, pesq_en_sum/len_list, (pesq_en_sum-pesq_raw_sum)/len_list))
    f.write('SDR_raw:%.3f, SDR_en:%.3f, SDRi_avg:%.3f. \r\n' % (sdr_raw_sum/len_list, sdr_en_sum/len_list, (sdr_en_sum-sdr_raw_sum)/len_list))
    f.write('STOI_raw:%.3f, STOI_en:%.3f, STOIi_avg:%.3f. \r\n' % (stoi_raw_sum/len_list, stoi_en_sum/len_list, (stoi_en_sum-stoi_raw_sum)/len_list))
  print('\n\n\n-----------------------------------------')
  print('PESQ_raw:%.3f, PESQ_en:%.3f, PESQi_avg:%.3f. \r\n' % (pesq_raw_sum/len_list, pesq_en_sum/len_list, (pesq_en_sum-pesq_raw_sum)/len_list))
  print('SDR_raw:%.3f, SDR_en:%.3f, SDRi_avg:%.3f. \r\n' % (sdr_raw_sum/len_list, sdr_en_sum/len_list, (sdr_en_sum-sdr_raw_sum)/len_list))
  print('STOI_raw:%.3f, STOI_en:%.3f, STOIi_avg:%.3f. \r\n' % (stoi_raw_sum/len_list, stoi_en_sum/len_list, (stoi_en_sum-stoi_raw_sum)/len_list))
  sys.stdout.flush()


if __name__=='__main__':
  ckpt= NNET_PARAM.CHECK_POINT # don't forget to change FLAGS.PARAM
  decode_ans_file = os.path.join(NNET_PARAM.SAVE_DIR,'decode_'+ckpt)
  if not os.path.exists(decode_ans_file):
    os.makedirs(decode_ans_file)
  sess, model = build_session(ckpt, 1)

  if len(sys.argv)<=1:
    decode_file_list_8k = [
        'exp/rnn_irm/8k/s_2_00_MIX_1_clapping_8k.wav',
        'exp/rnn_irm/8k/s_8_01_MIX_4_rainning_8k.wav',
        'exp/rnn_irm/8k/s_8_21_MIX_3_factory_8k.wav',
        'exp/rnn_irm/8k/s_2_00_8k_raw.wav',
        'exp/rnn_irm/8k/s_8_01_8k_raw.wav',
        'exp/rnn_irm/8k/s_8_21_8k_raw.wav',
        'exp/rnn_irm/8k/speech1_8k.wav',
        'exp/rnn_irm/8k/speech5_8k.wav',
        'exp/rnn_irm/8k/speech6_8k.wav',
        'exp/rnn_irm/8k/speech7_8k.wav',
        'exp/real_test_fair/863_min/mixed_wav/863_1_8k_MIX_1_airplane.wav',
        # 'exp/rnn_irm/decode_nnet_C001_3/nnet_C001_3_007_speech7_8k.wav'
    ]

    decode_file_list_16k = [
        'exp/rnn_irm/16k/s_2_00_MIX_1_clapping_16k.wav',
        'exp/rnn_irm/16k/s_8_01_MIX_4_rainning_16k.wav',
        'exp/rnn_irm/16k/s_8_21_MIX_3_factory_16k.wav',
        'exp/rnn_irm/16k/s_2_00_16k_raw.wav',
        'exp/rnn_irm/16k/s_8_01_16k_raw.wav',
        'exp/rnn_irm/16k/s_8_21_16k_raw.wav',
        'exp/rnn_irm/16k/speech0_16k.wav',
        'exp/rnn_irm/16k/speech1_16k.wav',
        'exp/rnn_irm/16k/speech6_16k.wav',
        'exp/rnn_irm/16k/speech7_16k.wav',
        'exp/rnn_irm/16k/863_1_16k_MIX_1_airplane.wav',
    ]
    if MIXED_AISHELL_PARAM.FS == 8000:
      decode_file_list = decode_file_list_8k
    elif MIXED_AISHELL_PARAM.FS == 16000:
      decode_file_list = decode_file_list_16k
    else:
      print('PARAM.FS error, exit.'),exit(-1)
    for i, mixed_dir in enumerate(decode_file_list):
      print(i+1,mixed_dir)
      waveData, sr = audio_tool.read_audio(mixed_dir)
      reY, mask = decode_one_wav(sess,model,waveData)
      print(np.max(reY))
      abs_max = (2 ** (MIXED_AISHELL_PARAM.AUDIO_BITS - 1) - 1)
      reY = np.where(reY>abs_max,abs_max,reY)
      reY = np.where(reY<-abs_max,-abs_max,reY)
      audio_tool.write_audio(os.path.join(decode_ans_file,
                                          (ckpt+'_%03d_' % (i+1))+mixed_dir[mixed_dir.rfind('/')+1:]),
                             reY,
                             sr)
      file_name = mixed_dir[mixed_dir.rfind('/')+1:mixed_dir.rfind('.')]
      spectrum_tool.picture_spec(mask,
                                 os.path.join(decode_ans_file,
                                              (ckpt+'_%03d_' % (i+1))+file_name))
  elif int(sys.argv[1])==0: # decode exp/test_oc
    mixed_dir = 'exp/test_oc/mixed_wav'
    decode_file_list = os.listdir(mixed_dir)
    decode_file_list = [os.path.join(mixed_dir,mixed) for mixed in decode_file_list]
    decode_file_list.sort()

    ref_dir = 'exp/test_oc/refer_wav'
    ref_list = os.listdir(ref_dir)
    ref_list = [os.path.join(ref_dir,ref) for ref in ref_list]
    ref_list.sort()

    # for mixed in decode_file_list:
    #   if mixed.find('clean') != -1 or mixed.find('ref.wav')!=-1:
    #     os.remove(mixed)

    # for ref in ref_list:
    #   if ref.find('clean') != -1 or ref.find('mixed.wav')!=-1:
    #     os.remove(ref)

    decode_and_getMeature(decode_file_list, ref_list, sess, model, decode_ans_file, False, 'test_oc.txt')

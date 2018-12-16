import tensorflow as tf
import numpy as np
import librosa
import os
import shutil
import time
import multiprocessing
import copy
import scipy.io
import datetime
import wave
from utils import spectrum_tool
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM

FILE_NAME = __file__[max(__file__.rfind('/')+1, 0):__file__.rfind('.')]
# region define
DATA_DICT_DIR = MIXED_AISHELL_PARAM.DATA_DICT_DIR
RAW_DATA = MIXED_AISHELL_PARAM.RAW_DATA
TFRECORD_DIR = MIXED_AISHELL_PARAM.TFRECORDS_DIR
PROCESS_NUM_GENERATE_TFERCORD = MIXED_AISHELL_PARAM.PROCESS_NUM_GENERATE_TFERCORD
TFRECORDS_NUM = MIXED_AISHELL_PARAM.TFRECORDS_NUM
LOG_NORM_MAX = MIXED_AISHELL_PARAM.LOG_NORM_MAX
LOG_NORM_MIN = MIXED_AISHELL_PARAM.LOG_NORM_MIN
NFFT = MIXED_AISHELL_PARAM.NFFT
OVERLAP = MIXED_AISHELL_PARAM.OVERLAP
FS = MIXED_AISHELL_PARAM.FS
LEN_WAWE_PAD_TO = MIXED_AISHELL_PARAM.LEN_WAWE_PAD_TO
UTT_SEG_FOR_MIX = MIXED_AISHELL_PARAM.UTT_SEG_FOR_MIX
DATASET_NAMES = MIXED_AISHELL_PARAM.DATASET_NAMES
DATASET_SIZES = MIXED_AISHELL_PARAM.DATASET_SIZES
WAVE_NORM = MIXED_AISHELL_PARAM.WAVE_NORM
NOISE_DIR = MIXED_AISHELL_PARAM.NOISE_DIR
# endregion


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _ini_data(wave_dir, noise_dir, out_dir):
  data_dict_dir = out_dir
  if os.path.exists(data_dict_dir):
    shutil.rmtree(data_dict_dir)
  os.makedirs(data_dict_dir)
  clean_wav_speaker_set_dir = wave_dir
  os.makedirs(data_dict_dir+'/train')
  os.makedirs(data_dict_dir+'/validation')
  os.makedirs(data_dict_dir+'/test_cc')
  cwl_train_file = open(data_dict_dir+'/train/clean_wav_dir.list', 'a+')
  cwl_validation_file = open(
      data_dict_dir+'/validation/clean_wav_dir.list', 'a+')
  cwl_test_cc_file = open(data_dict_dir+'/test_cc/clean_wav_dir.list', 'a+')
  clean_wav_list_train = []
  clean_wav_list_validation = []
  clean_wav_list_test_cc = []
  speaker_list = os.listdir(clean_wav_speaker_set_dir)
  speaker_list.sort()
  for speaker_name in speaker_list:
    speaker_dir = clean_wav_speaker_set_dir+'/'+speaker_name
    if os.path.isdir(speaker_dir):
      speaker_wav_list = os.listdir(speaker_dir)
      speaker_wav_list.sort()
      for wav in speaker_wav_list[:UTT_SEG_FOR_MIX[0]]:
        # 清洗长度为0的数据
        if wav[-4:] == ".wav" and os.path.getsize(speaker_dir+'/'+wav) > 2048:
          cwl_train_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_train.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[UTT_SEG_FOR_MIX[0]:UTT_SEG_FOR_MIX[1]]:
        if wav[-4:] == ".wav" and os.path.getsize(speaker_dir+'/'+wav) > 2048:
          cwl_validation_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_validation.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[UTT_SEG_FOR_MIX[1]:]:
        if wav[-4:] == ".wav" and os.path.getsize(speaker_dir+'/'+wav) > 2048:
          cwl_test_cc_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_test_cc.append(speaker_dir+'/'+wav)

  cwl_train_file.close()
  cwl_validation_file.close()
  cwl_test_cc_file.close()
  print('train clean: '+str(len(clean_wav_list_train)))
  print('validation clean: '+str(len(clean_wav_list_validation)))
  print('test_cc clean: '+str(len(clean_wav_list_test_cc)))

  # NOISE LIST
  noise_wav_list = os.listdir(noise_dir)
  noise_wav_list = [os.path.join(noise_dir, noise) for noise in noise_wav_list]

  dataset_names = DATASET_NAMES
  dataset_mixedutt_num = DATASET_SIZES
  all_mixed = 0
  all_stime = time.time()
  for (clean_wav_list, j) in zip((clean_wav_list_train, clean_wav_list_validation, clean_wav_list_test_cc), range(3)):
    print('\n'+dataset_names[j]+" data preparing...")
    s_time = time.time()
    mixed_wav_list_file = open(
        data_dict_dir+'/'+dataset_names[j]+'/mixed_wav_dir.list', 'a+')
    mixed_wave_list = []
    len_wav_list = len(clean_wav_list)
    len_noise_wave_list = len(noise_wav_list)
    # print(len_wav_list,len_noise_wave_list)
    generated_num = 0
    while generated_num < dataset_mixedutt_num[j]:
      uttid = np.random.randint(len_wav_list)
      noiseid = np.random.randint(len_noise_wave_list)
      utt1_dir = clean_wav_list[uttid]
      utt2_dir = noise_wav_list[noiseid]
      generated_num += 1
      mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
      mixed_wave_list.append([utt1_dir, utt2_dir])
    # for i_utt in range(len_wav_list): # n^2混合，数据量巨大
    #   for j_utt in range(i_utt,len_wav_list):
    #     utt1_dir=clean_wav_list[i_utt]
    #     utt2_dir=clean_wav_list[j_utt]
    #     speaker1 = utt1_dir.split('/')[-2]
    #     speaker2 = utt2_dir.split('/')[-2]
    #     if speaker1 == speaker2:
    #       continue
    #     mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
    #     mixed_wave_list.append([utt1_dir, utt2_dir])
    mixed_wav_list_file.close()
    scipy.io.savemat(
        data_dict_dir+'/'+dataset_names[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
    all_mixed += len(mixed_wave_list)
    print(dataset_names[j]+' data preparation over, Mixed num: ' +
          str(len(mixed_wave_list))+(', Cost time %dS.') % (time.time()-s_time))
  print('\nData preparation over, all mixed num: %d,cost time: %dS' %
        (all_mixed, time.time()-all_stime))


def _get_waveData1_waveData2(file1, noise_file):
  f1 = wave.open(file1, 'rb')
  f2 = wave.open(noise_file, 'rb')
  waveData = np.fromstring(f1.readframes(f1.getnframes()),
                           dtype=np.int16)
  noiseData = np.fromstring(f2.readframes(f2.getnframes()),
                            dtype=np.int16)
  f1.close()
  f2.close()
  while len(waveData) < LEN_WAWE_PAD_TO:
    waveData = np.tile(waveData, 2)
  while len(noiseData) < LEN_WAWE_PAD_TO:
    noiseData = np.tile(noiseData, 2)

  len_noise = len(noiseData)
  noise_begin = np.random.randint(len_noise-LEN_WAWE_PAD_TO+1)
  waveData = waveData[:LEN_WAWE_PAD_TO]
  noiseData = noiseData[noise_begin:noise_begin+LEN_WAWE_PAD_TO]
  if WAVE_NORM:
    waveData = waveData/np.max(np.abs(waveData)) * 32767
    noiseData = noiseData/np.max(np.abs(noiseData)) * 32767
  return waveData, noiseData


def _mix_wav(waveData1, waveData2):
  # 混合语音
  mixedData = (waveData1+waveData2)/2
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据
  return mixedData


def rmNormalization(tmp):
  tmp = (10**(tmp*(LOG_NORM_MAX-LOG_NORM_MIN)+LOG_NORM_MIN))-0.5
  ans = np.where(tmp > 0, tmp, 0)  # 防止计算误差导致的反归一化结果为负数
  return ans


def _extract_norm_log_mag_spec(data):
  # 归一化的幅度谱对数
  mag_spec = spectrum_tool.magnitude_spectrum_librosa_stft(
      data, NFFT, OVERLAP)
  # Normalization
  log_mag_spec = np.log10(mag_spec+0.5)
  # #TODO
  # print('???', np.max(log_mag_spec), np.min(log_mag_spec))
  log_mag_spec[log_mag_spec > LOG_NORM_MAX] = LOG_NORM_MAX
  log_mag_spec[log_mag_spec < LOG_NORM_MIN] = LOG_NORM_MIN
  log_mag_spec += np.abs(LOG_NORM_MIN)
  log_mag_spec /= (np.abs(LOG_NORM_MIN)+LOG_NORM_MAX)
  # mean=np.mean(log_mag_spec)
  # var=np.var(log_mag_spec)
  # log_mag_spec=(log_mag_spec-mean)/var
  return log_mag_spec


def _extract_phase(data):
  theta = spectrum_tool.phase_spectrum_librosa_stft(data, NFFT, OVERLAP)
  return theta


def _extract_feature_x_y_xtheta_ytheta(utt_dir1, utt_dir2):
  waveData1, waveData2 = _get_waveData1_waveData2(
      utt_dir1, utt_dir2)
  # utt2作为噪音
  noise_rate = np.random.random()
  mixedData = _mix_wav(waveData1, waveData2*noise_rate)

  # nchannels = 1
  # sampwidth = 2  # 采样位宽，2表示16位
  # framerate = 16000
  # nframes = len(mixedData)
  # comptype = "NONE"
  # compname = "not compressed"
  # name1 = utt_dir1[utt_dir1.rfind('/')+1:utt_dir1.rfind('.')]
  # name2 = utt_dir2[utt_dir2.rfind('/')+1:]
  # wavefile = wave.open('mixwave/mixed_'+name1+"_"+name2, 'wb')
  # wavefile.setparams((nchannels, sampwidth, framerate, nframes,
  #                     comptype, compname))
  # wavefile.writeframes(
  #     np.array(mixedData, dtype=np.int16))

  X = _extract_norm_log_mag_spec(mixedData)
  Y = _extract_norm_log_mag_spec(waveData1)
  x_theta = _extract_phase(mixedData)
  y_theta = _extract_phase(waveData1)

  return [X, Y, x_theta, y_theta]


def parse_func(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
      'xtheta': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'ytheta': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
  }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels'], 0, 0, length


def parse_func_with_theta(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
      'xtheta': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'ytheta': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
  }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels'], sequence['xtheta'], sequence['ytheta'], length


def _gen_tfrecord_minprocess(
        dataset_index_list, s_site, e_site, dataset_dir, i_process):
  tfrecord_savedir = os.path.join(dataset_dir, ('%08d.tfrecords' % i_process))
  with tf.python_io.TFRecordWriter(tfrecord_savedir) as writer:
    for i in range(s_site, e_site):
      index_ = dataset_index_list[i]
      X_Y_Xtheta_Ytheta = _extract_feature_x_y_xtheta_ytheta(index_[
                                                             0], index_[1])
      X = np.reshape(np.array(X_Y_Xtheta_Ytheta[0], dtype=np.float32),
                     newshape=[-1, NNET_PARAM.INPUT_SIZE])
      Y = np.reshape(np.array(X_Y_Xtheta_Ytheta[1], dtype=np.float32),
                     newshape=[-1, NNET_PARAM.OUTPUT_SIZE])
      Xtheta = np.reshape(np.array(X_Y_Xtheta_Ytheta[2], dtype=np.float32),
                          newshape=[-1, NNET_PARAM.INPUT_SIZE])
      Ytheta = np.reshape(np.array(X_Y_Xtheta_Ytheta[3], dtype=np.float32),
                          newshape=[-1, NNET_PARAM.OUTPUT_SIZE])
      input_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=input_))
          for input_ in X]
      label_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=label))
          for label in Y]
      xtheta_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=xtheta))
          for xtheta in Xtheta]
      ytheta_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=ytheta))
          for ytheta in Ytheta]
      feature_list = {
          'inputs': tf.train.FeatureList(feature=input_features),
          'labels': tf.train.FeatureList(feature=label_features),
          'xtheta': tf.train.FeatureList(feature=xtheta_features),
          'ytheta': tf.train.FeatureList(feature=ytheta_features),
      }
      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      record = tf.train.SequenceExample(feature_lists=feature_lists)
      writer.write(record.SerializeToString())
    writer.flush()
    # print(dataset_dir + ('/%08d.tfrecords' % i), 'write done')


def generate_tfrecord(gen=True):
  tfrecords_dir = TFRECORD_DIR
  train_tfrecords_dir = os.path.join(tfrecords_dir, 'train')
  val_tfrecords_dir = os.path.join(tfrecords_dir, 'validation')
  testcc_tfrecords_dir = os.path.join(tfrecords_dir, 'test_cc')
  dataset_dir_list = [train_tfrecords_dir,
                      val_tfrecords_dir,
                      testcc_tfrecords_dir]

  if gen:
    _ini_data(RAW_DATA, NOISE_DIR, DATA_DICT_DIR)
    if os.path.exists(train_tfrecords_dir):
      shutil.rmtree(train_tfrecords_dir)
    if os.path.exists(val_tfrecords_dir):
      shutil.rmtree(val_tfrecords_dir)
    if os.path.exists(testcc_tfrecords_dir):
      shutil.rmtree(testcc_tfrecords_dir)
    os.makedirs(train_tfrecords_dir)
    os.makedirs(val_tfrecords_dir)
    os.makedirs(testcc_tfrecords_dir)

    gen_start_time = time.time()
    for dataset_dir in dataset_dir_list:
      start_time = time.time()
      dataset_index_list = None
      if dataset_dir[-2:] == 'in':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/train/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'on':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/validation/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'cc':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/test_cc/mixed_wav_dir.mat')["mixed_wav_dir"]

      # 使用.mat，字符串长度会强制对齐，所以去掉空格
      dataset_index_list = [[index_[0].replace(' ', ''),
                             index_[1].replace(' ', '')] for index_ in dataset_index_list]
      len_dataset = len(dataset_index_list)
      minprocess_utt_num = int(
          len_dataset/TFRECORDS_NUM)
      pool = multiprocessing.Pool(PROCESS_NUM_GENERATE_TFERCORD)
      for i_process in range(TFRECORDS_NUM):
        s_site = i_process*minprocess_utt_num
        e_site = s_site+minprocess_utt_num
        if i_process == (PROCESS_NUM_GENERATE_TFERCORD-1):
          e_site = len_dataset
        # print(s_site,e_site)
        pool.apply_async(_gen_tfrecord_minprocess,
                         (dataset_index_list,
                          s_site,
                          e_site,
                          dataset_dir,
                          i_process))
        # _gen_tfrecord_minprocess(dataset_index_list,
        #                          s_site,
        #                          e_site,
        #                          dataset_dir,
        #                          i_process)
      pool.close()
      pool.join()

      print(dataset_dir+' set extraction over. cost time %06dS' %
            (time.time()-start_time))
    print('Generate TFRecord over. cost time %06dS' %
          (time.time()-gen_start_time))

  train_set = os.path.join(train_tfrecords_dir, '*.tfrecords')
  val_set = os.path.join(val_tfrecords_dir, '*.tfrecords')
  testcc_set = os.path.join(testcc_tfrecords_dir, '*.tfrecords')
  return train_set, val_set, testcc_set


def get_batch_use_tfdata2(tfrecords_list, get_theta=False):
  files = tf.data.Dataset.list_files(tfrecords_list)
  files = files.take(MIXED_AISHELL_PARAM.MAX_TFRECORD_FILES)
  if MIXED_AISHELL_PARAM.SHUFFLE:
    files = files.shuffle(MIXED_AISHELL_PARAM.PROCESS_NUM_GENERATE_TFERCORD)
  if not MIXED_AISHELL_PARAM.SHUFFLE:
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=1,
                               block_length=NNET_PARAM.batch_size,
                               #  num_parallel_calls=1,
                               )
  else:  # shuffle
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=NNET_PARAM.batch_size*3,
                               #  block_length=1,
                               num_parallel_calls=NNET_PARAM.num_threads_processing_data,
                               )
  if MIXED_AISHELL_PARAM.SHUFFLE:
    dataset = dataset.shuffle(NNET_PARAM.batch_size*3)
  # region
  # !tf.data with tf.device(cpu) OOM???
  # dataset = dataset.map(
  #     map_func=parse_func,
  #     num_parallel_calls=NNET_PARAM.num_threads_processing_data)
  # dataset = dataset.padded_batch(
  #     NNET_PARAM.batch_size,
  #     padded_shapes=([None, NNET_PARAM.INPUT_SIZE],
  #                    [None, NNET_PARAM.OUTPUT_SIZE],
  #                    [None, NNET_PARAM.OUTPUT_SIZE],
  #                    []))
  # endregion
  # !map_and_batch efficient is better than map+paded_batch
  dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=parse_func_with_theta if get_theta else parse_func,
      batch_size=NNET_PARAM.batch_size,
      num_parallel_calls=NNET_PARAM.num_threads_processing_data,
      # num_parallel_batches=2,
  ))
  # dataset = dataset.prefetch(buffer_size=NNET_PARAM.batch_size) # perfetch 太耗内存，并没有明显的速度提升
  dataset_iter = dataset.make_initializable_iterator()
  x_batch, y_batch, xtheta, ytheta, lengths_batch = dataset_iter.get_next()
  return x_batch, y_batch, xtheta, ytheta, lengths_batch, dataset_iter


def get_batch_use_tfdata(tfrecords_list, get_theta=False):
  files = os.listdir(tfrecords_list[:-11])
  files = files[:min(MIXED_AISHELL_PARAM.MAX_TFRECORD_FILES, len(files))]
  files = [os.path.join(tfrecords_list[:-11], file) for file in files]
  dataset_list = [tf.data.TFRecordDataset(file).map(parse_func_with_theta if get_theta else parse_func,
                                                    num_parallel_calls=NNET_PARAM.num_threads_processing_data) for file in files]

  num_classes = MIXED_AISHELL_PARAM.MAX_TFRECORD_FILES
  num_classes_per_batch = MIXED_AISHELL_PARAM.MAX_TFRECORD_FILES
  num_utt_per_class = NNET_PARAM.batch_size//num_classes_per_batch

  def generator(_):
    # Sample `num_classes_per_batch` classes for the batch
    sampled = tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
    # Repeat each element `num_images_per_class` times
    batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_utt_per_class])
    return tf.to_int64(tf.reshape(batch_labels, [-1]))

  selector = tf.contrib.data.Counter().map(generator)
  selector = selector.apply(tf.contrib.data.unbatch())

  dataset = tf.data.experimental.choose_from_datasets(dataset_list, selector)
  dataset = dataset.batch(num_classes_per_batch * num_utt_per_class)
  # dataset = dataset.prefetch(buffer_size=NNET_PARAM.batch_size) # perfetch 太耗内存，并没有明显的速度提升
  dataset_iter = dataset.make_initializable_iterator()
  x_batch, y_batch, xtheta, ytheta, lengths_batch = dataset_iter.get_next()
  return x_batch, y_batch, xtheta, ytheta, lengths_batch, dataset_iter

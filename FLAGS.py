class NNET_PARAM:
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  MASK_TYPE = "IRM"  # 'PSIRM'
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  decode = False
  batch_size = 128
  learning_rate = 0.001
  start_halving_impr = 0.0003
  resume_training = 'false'
  start_epoch = 0
  min_epochs = 10  # Min number of epochs to run trainer without halving.
  max_epochs = 50  # Max number of epochs to run trainer totally.
  halving_factor = 0.7  # Factor for halving.
  # Halving when ralative loss is lower than start_halving_impr.
  start_halving_impr = 0.003
  # Stop when relative loss is lower than end_halving_impr.
  end_halving_impr = 0.0005
  # The num of threads to read tfrecords files.
  num_threads_processing_data = 64
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'
  GRIFFIN_ITERNUM = 50

  GPU_RAM_ALLOW_GROWTH = True
  USE_MULTIGPU = False  # dont't use multiGPU,because it is not work now...
  GPU_LIST = [0, 2]
  if USE_MULTIGPU:
    if batch_size % len(GPU_LIST) == 0:
      batch_size //= len(GPU_LIST)
    else:
      print('Batch_size %d cannot divided by gpu num %d.' %
            (batch_size, len(GPU_LIST)))
      exit(-1)

  minibatch_size = 400  # batch num to show


class MIXED_AISHELL_PARAM:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  # RAW_DATA = '/aishell_90_speaker' # for docker
  RAW_DATA = '/home/student/work/pit_test/data'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 320
  SHUFFLE = False

  # TFRECORDS_DIR = '/feature_tfrecords_utt03s_irm' # for docker
  TFRECORDS_DIR = '/home/student/work/lhf/alldata/irm-data/feature_tfrecords_utt03s_irm'
  # 'big' or 'small'.if 'small', one file per record.
  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [260, 290]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [700000, 18000, 180000]

  WAVE_NORM = True
  MAX_NOISE_RATE = 1.0  # wave_norm + noise_rate
  NOISE_DIR = '/home/student/work/lhf/noise_lhf'
  # WAVE_NORM = False
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -3
  NFFT = 512
  OVERLAP = 256
  FS = 16000

  MAX_TFRECORD_FILES=320 # 640

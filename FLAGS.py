from losses import loss

class C3_NNET:
  MASK_TYPE = "IRM"  # or 'PSIRM'
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE # "MSE" "MSE_LOW_FS_IMPROVE"
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  '''
  decode:
    decode by the flod '_decode_index'. one set per (.list) file.
  '''
  decode = 1  # 0:train; 1:decode_for_show; 2:decode_test_set_calculate_SDR_Improvement

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
  num_threads_processing_data = 16
  decode_output_speaker_volume_amp = True
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'.
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

class C3_DATA:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  #RAW_DATA = '/all_data/aishell2_speaker_list' # for docker
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = True
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  SHUFFLE = False

  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [400, 460]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  MAX_TFRECORD_FILES_USED=160 # <=TFRECORDS_NUM

  #NOISE_DIR = '/all_data/many_noise' # for docker
  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  NFFT = 512
  OVERLAP = NFFT - 256
  FS = 16000
  FEATURE_TYPE = 'LOG_MAG'  # MAG or LOG_MAG
  MASK_ON_MAG_EVEN_LOGMAG = None
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -3
  MAG_NORM_MAX = 1e6
  MAG_NORM_MIN = 0


  AUDIO_VOLUME_AMP=True


  MIX_METHOD = 'LINEAR' # "LINEAR" "SNR"
  MAX_SNR = 9  # 以不同信噪比混合
  MIN_SNR = -6
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  # TFRECORDS_DIR = '/all_data/feature_tfrecords' # for docker
  TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm'


class C6_NNET:
  MASK_TYPE = "IRM"  # or 'PSIRM'
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE # "MSE" "MSE_LOW_FS_IMPROVE"
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  '''
  decode:
    decode by the flod '_decode_index'. one set per (.list) file.
  '''
  decode = 1  # 0:train; 1:decode_for_show; 2:decode_test_set_calculate_SDR_Improvement

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
  num_threads_processing_data = 16
  decode_output_speaker_volume_amp = True
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'.
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

class C6_DATA:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  # RAW_DATA = '/aishell_90_speaker' # for docker
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  SHUFFLE = False

  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [400, 460]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  MAX_TFRECORD_FILES_USED=160 # <=TFRECORDS_NUM

  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  NFFT = 512
  OVERLAP = NFFT - 256
  FS = 16000
  FEATURE_TYPE = 'LOG_MAG'  # MAG or LOG_MAG
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -0.3
  MAG_NORM_MAX = 1e6
  MAG_NORM_MIN = 0


  AUDIO_VOLUME_AMP=True


  MIX_METHOD = 'LINEAR' # "LINEAR" "SNR"
  MAX_SNR = 9  # 以不同信噪比混合
  MIN_SNR = -6
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  # TFRECORDS_DIR = '/feature_tfrecords_utt03s_irm' # for docker
  TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm'
  if MIX_METHOD =='SNR':
    TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm_SNR_MIX'

class C7_NNET:
  MASK_TYPE = "IRM"  # or 'PSIRM'
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE # "MSE" "MSE_LOW_FS_IMPROVE"
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  '''
  decode:
    decode by the flod '_decode_index'. one set per (.list) file.
  '''
  decode = 1  # 0:train; 1:decode_for_show; 2:decode_test_set_calculate_SDR_Improvement

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
  num_threads_processing_data = 16
  decode_output_speaker_volume_amp = True
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'.
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

class C7_DATA:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  # RAW_DATA = '/aishell_90_speaker' # for docker
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  SHUFFLE = False

  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [400, 460]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  MAX_TFRECORD_FILES_USED=160 # <=TFRECORDS_NUM

  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  NFFT = 512
  OVERLAP = NFFT - 256
  FS = 16000
  FEATURE_TYPE = 'MAG'  # MAG or LOG_MAG
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -0.3
  MAG_NORM_MAX = 1e6
  MAG_NORM_MIN = 0


  AUDIO_VOLUME_AMP=True


  MIX_METHOD = 'LINEAR' # "LINEAR" "SNR"
  MAX_SNR = 9  # 以不同信噪比混合
  MIN_SNR = -6
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  # TFRECORDS_DIR = '/feature_tfrecords_utt03s_irm' # for docker
  if FEATURE_TYPE == "LOG_MAG":
    TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm'
    if MIX_METHOD =='SNR':
      TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm_SNR_MIX'
  elif FEATURE_TYPE == "MAG":
    TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_mag_utt03s_irm'
    if MIX_METHOD =='SNR':
      TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_mag_utt03s_irm_SNR_MIX'

class C8_NNET:
  MASK_TYPE = "PSIRM"  # or 'PSIRM'
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE # "MSE" "MSE_LOW_FS_IMPROVE"
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  '''
  decode:
    decode by the flod '_decode_index'. one set per (.list) file.
  '''
  decode = 1  # 0:train; 1:decode_for_show; 2:decode_test_set_calculate_SDR_Improvement

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
  num_threads_processing_data = 16
  decode_output_speaker_volume_amp = False
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'.
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

class C8_DATA:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  # RAW_DATA = '/aishell_90_speaker' # for docker
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  SHUFFLE = False

  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [400, 460]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  MAX_TFRECORD_FILES_USED=160 # <=TFRECORDS_NUM

  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  NFFT = 512
  OVERLAP = NFFT - 256
  FS = 16000
  FEATURE_TYPE = 'MAG'  # MAG or LOG_MAG
  MASK_ON_MAG_EVEN_LOGMAG = None # It worked only when FEATURE_TYPE is 'MAG'.
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -0.3
  MAG_NORM_MAX = 1e6
  MAG_NORM_MIN = 0

  AUDIO_VOLUME_AMP=False

  MIX_METHOD = 'LINEAR' # "LINEAR" "SNR"
  MAX_SNR = 9  # 以不同信噪比混合
  MIN_SNR = -6
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  # TFRECORDS_DIR = '/feature_tfrecords_utt03s_irm' # for docker
  if FEATURE_TYPE == "LOG_MAG":
    TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm'
    if MIX_METHOD =='SNR':
      TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm_SNR_MIX'
  elif FEATURE_TYPE == "MAG":
    TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_mag_utt03s_irm'
    if MIX_METHOD =='SNR':
      TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_mag_utt03s_irm_SNR_MIX'

class C9_NNET:
  MASK_TYPE = "PSIRM"  # or 'PSIRM'
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE # "MSE" "MSE_LOW_FS_IMPROVE"
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  INPUT_SIZE = 513
  OUTPUT_SIZE = 513
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  '''
  decode:
    decode by the flod '_decode_index'. one set per (.list) file.
  '''
  decode = 1  # 0:train; 1:decode_for_show; 2:decode_test_set_calculate_SDR_Improvement

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
  num_threads_processing_data = 16
  decode_output_speaker_volume_amp = False
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'.
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

class C9_DATA:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  # RAW_DATA = '/aishell_90_speaker' # for docker
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  SHUFFLE = False

  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [400, 460]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  MAX_TFRECORD_FILES_USED=160 # <=TFRECORDS_NUM

  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  NFFT = 1024
  OVERLAP = 1024-160
  FS = 16000
  FEATURE_TYPE = 'MAG'  # MAG or LOG_MAG
  MASK_ON_MAG_EVEN_LOGMAG = None # It worked only when FEATURE_TYPE is 'MAG'.
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -0.3
  MAG_NORM_MAX = 1e6
  MAG_NORM_MIN = 0

  AUDIO_VOLUME_AMP=False

  MIX_METHOD = 'LINEAR' # "LINEAR" "SNR"
  MAX_SNR = 9  # 以不同信噪比混合
  MIN_SNR = -6
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  # TFRECORDS_DIR = '/feature_tfrecords_utt03s_irm' # for docker
  TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_mag_utt03s_irm_1024fft'

class C10_NNET:
  MASK_TYPE = "PSIRM"  # or 'PSIRM'
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE # "MSE" "MSE_LOW_FS_IMPROVE"
  MODEL_TYPE = 'BLSTM'  # 'BLSTM' or 'BGRU'
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_irm'
  '''
  decode:
    decode by the flod '_decode_index'. one set per (.list) file.
  '''
  decode = 1  # 0:train; 1:decode_for_show; 2:decode_test_set_calculate_SDR_Improvement

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
  num_threads_processing_data = 16
  decode_output_speaker_volume_amp = False
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','CLEANED','GRIFFIN_LIM'.
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

class C10_DATA:
  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  # RAW_DATA = '/aishell_90_speaker' # for docker
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  SHUFFLE = False

  LEN_WAWE_PAD_TO = 16000*3  # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [400, 460]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  MAX_TFRECORD_FILES_USED=160 # <=TFRECORDS_NUM

  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  NFFT = 1024
  OVERLAP = NFFT - 160
  FS = 16000
  FEATURE_TYPE = 'LOG_MAG'  # MAG or LOG_MAG
  MASK_ON_MAG_EVEN_LOGMAG = True
  LOG_NORM_MAX = 6
  LOG_NORM_MIN = -0.3
  MAG_NORM_MAX = 1e6
  MAG_NORM_MIN = 0

  AUDIO_VOLUME_AMP=False

  MIX_METHOD = 'LINEAR' # "LINEAR" "SNR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  # TFRECORDS_DIR = '/feature_tfrecords_utt03s_irm' # for docker
  TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_logmag_utt03s_irm_1024fft'

NNET_PARAM = C3_NNET
MIXED_AISHELL_PARAM = C3_DATA

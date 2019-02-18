import time
import tensorflow as tf
import numpy as np
import sys
import utils
import utils.audio_tool
import os
import shutil
from models.lstm_SE import SE_MODEL
import wave
import gc
from dataManager import mixed_aishell_tfrecord_io as wav_tool
from dataManager.mixed_aishell_tfrecord_io import rmNormalization
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM
from dataManager.mixed_aishell_tfrecord_io import generate_tfrecord, get_batch_use_tfdata

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]


def decode_testSet_get_SDR_Impr():
  pass


def show_onewave(decode_ans_dir, name, x_spec, y_spec, x_angle, y_angle, cleaned, mixed_wav, mask):
  # show the 5 data.(wav,spec,sound etc.)
  x_spec = np.array(rmNormalization(x_spec))
  if MIXED_AISHELL_PARAM.FEATURE_TYPE == 'LOG_MAG' and MIXED_AISHELL_PARAM.MASK_ON_MAG_EVEN_LOGMAG:
    cleaned = np.array(cleaned)
  else:
    cleaned = np.array(rmNormalization(cleaned))
  # 去噪阈值 #TODO
  # cleaned = np.where(cleaned > 300, cleaned, cleaned/10)
  y_spec = np.array(rmNormalization(y_spec))

  # wav_spec(spectrum)
  utils.spectrum_tool.picture_spec(np.log10(cleaned+MIXED_AISHELL_PARAM.LOG_BIAS),
                                   decode_ans_dir+'/restore_spec_'+name)
  utils.spectrum_tool.picture_spec(np.log10(x_spec+MIXED_AISHELL_PARAM.LOG_BIAS),
                                   decode_ans_dir+'/mixed_spec_'+name)
  # TODO RM
  # x_spec_bak=x_spec
  utils.spectrum_tool.picture_spec(np.log10(y_spec+MIXED_AISHELL_PARAM.LOG_BIAS),
                                   decode_ans_dir+'/raw_spec_'+name)

  # mask_spec
  utils.spectrum_tool.picture_spec(mask,
                                   decode_ans_dir+'/mask_'+name)
  x_spec = x_spec * np.exp(x_angle*1j)
  y_spec = y_spec * np.exp(y_angle*1j)
  if NNET_PARAM.RESTORE_PHASE == 'CLEANED':
    cleaned_spec = cleaned * np.exp(y_angle*1j)
  elif NNET_PARAM.RESTORE_PHASE == 'MIXED':
    cleaned_spec = cleaned * np.exp(x_angle*1j)
  elif NNET_PARAM.RESTORE_PHASE == 'GRIFFIN_LIM':
    cleaned_spec = utils.spectrum_tool.griffin_lim(cleaned, mixed_wav,
                                                   MIXED_AISHELL_PARAM.NFFT,
                                                   MIXED_AISHELL_PARAM.OVERLAP,
                                                   NNET_PARAM.GRIFFIN_ITERNUM)

  framerate = MIXED_AISHELL_PARAM.FS
  bits = 16

  # write restore wave
  reY = utils.spectrum_tool.librosa_istft(
      cleaned_spec, MIXED_AISHELL_PARAM.NFFT, MIXED_AISHELL_PARAM.OVERLAP)
  if NNET_PARAM.decode_output_speaker_volume_amp:  # norm resotred wave
    reY = reY/np.max(np.abs(reY))*32767
  utils.audio_tool.write_audio(decode_ans_dir+'/restore_audio_'+name+'.wav',
                               reY,
                               framerate,
                               bits, 'wav')

  # write raw wave
  rawY = utils.spectrum_tool.librosa_istft(
      y_spec, MIXED_AISHELL_PARAM.NFFT, MIXED_AISHELL_PARAM.OVERLAP)
  utils.audio_tool.write_audio(decode_ans_dir+'/raw_audio_'+name+'.wav',
                               rawY,
                               framerate,
                               bits, 'wav')

  # # write mixed wave
  mixedWave = utils.spectrum_tool.librosa_istft(
      x_spec, MIXED_AISHELL_PARAM.NFFT, MIXED_AISHELL_PARAM.OVERLAP)
  utils.audio_tool.write_audio(decode_ans_dir+'/mixed_audio_'+name+'.wav',
                               mixedWave,
                               framerate,
                               bits, 'wav')

  # wav_pic(oscillograph)
  utils.spectrum_tool.picture_wave(reY,
                                   decode_ans_dir +
                                   '/restore_wav_'+name,
                                   framerate)
  # TODO RM
  # spec = utils.spectrum_tool.magnitude_spectrum_librosa_stft(reY,512,256)
  # utils.spectrum_tool.picture_spec(np.log10(DATA_PARAM.LOG_BIAS+spec),decode_ans_dir+'/C7_fu0.3_iter7_')
  # spec_subs = spec - x_spec_bak
  # utils.spectrum_tool.picture_spec(np.log10(DATA_PARAM.LOG_BIAS+spec_subs),decode_ans_dir+'/error')
  # print(np.max(spec_subs[100:]), np.min(spec_subs[100:]))
  # print(np.shape(spec_subs))
  # utils.spectrum_tool.picture_spec(spec_subs,decode_ans_dir+'/log_error')
  #
  utils.spectrum_tool.picture_wave(rawY,
                                   decode_ans_dir +
                                   '/raw_wav_' + name,
                                   framerate)



def decode_oneset_old(setname, set_index_list_dir, ckpt_dir='nnet'):
  dataset_index_file = open(set_index_list_dir, 'r')
  dataset_index_strlist = dataset_index_file.readlines()
  if len(dataset_index_strlist) <= 0:
    print('Set %s have no element.' % setname)
    return
  x_spec = []
  y_spec = []
  x_theta = []
  y_theta = []
  lengths = []
  mixed_wave = []
  for i, index_str in enumerate(dataset_index_strlist):
    uttdir1, uttdir2 = index_str.replace('\n', '').split(' ')
    # print(uttdir1,uttdir2)
    uttwave1, uttwave2 = wav_tool._get_waveData1_waveData2_MAX_Volume(
        uttdir1, uttdir2)
    if uttdir2 != 'None':  # 将帧级语音和噪音混合后解码
      if MIXED_AISHELL_PARAM.MIX_METHOD == 'SNR':
        mixed_wave_t = wav_tool._mix_wav_by_SNR(uttwave1, uttwave2)
      if MIXED_AISHELL_PARAM.MIX_METHOD == 'LINEAR':
        mixed_wave_t = wav_tool._mix_wav_by_LINEAR(uttwave1, uttwave2)
      # mixed_wave_t = wav_tool._mix_wav_by_SNR(uttwave1, uttwave2)
    else:  # 解码单一混合语音（uttwave1是带有噪声的语音）
      mixed_wave_t = uttwave1
    x_spec_t = wav_tool._extract_norm_log_mag_spec(mixed_wave_t)
    y_spec_t = wav_tool._extract_norm_log_mag_spec(uttwave1)
    x_theta_t = wav_tool._extract_phase(mixed_wave_t)
    y_theta_t = wav_tool._extract_phase(uttwave1)
    mixed_wave.append(mixed_wave_t)
    x_spec.append(x_spec_t)
    y_spec.append(y_spec_t)
    x_theta.append(x_theta_t)
    y_theta.append(y_theta_t)
    lengths.append(np.shape(x_spec_t)[0])

  #  multi_single_mixed_wave_test
  max_length = np.max(lengths)
  x_spec = [np.pad(x_spec_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
            for i, x_spec_t in enumerate(x_spec)]
  y_spec = [np.pad(y_spec_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
            for i, y_spec_t in enumerate(y_spec)]
  x_theta = [np.pad(x_theta_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
             for i, x_theta_t in enumerate(x_theta)]
  y_theta = [np.pad(y_theta_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
             for i, y_theta_t in enumerate(y_theta)]

  x_spec = np.array(x_spec, dtype=np.float32)
  y_spec = np.array(y_spec, dtype=np.float32)
  x_theta = np.array(x_theta, dtype=np.float32)
  y_theta = np.array(y_theta, dtype=np.float32)
  lengths = np.array(lengths, dtype=np.int32)


  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_spec, y_spec, x_theta, y_theta, lengths))
        dataset = dataset.batch(NNET_PARAM.batch_size)
        dataset_iter = dataset.make_one_shot_iterator()
        # dataset_iter = dataset.make_initializable_iterator()
        x_batch, y_batch, x_theta_batch, y_theta_batch, lengths_batch = dataset_iter.get_next()

    with tf.name_scope('model'):
      model = SE_MODEL(x_batch, y_batch, lengths_batch, x_theta_batch, y_theta_batch,
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

  decode_ans_dir = os.path.join(
      NNET_PARAM.SAVE_DIR, 'decode_ans', setname)
  if os.path.exists(decode_ans_dir):
    shutil.rmtree(decode_ans_dir)
  os.makedirs(decode_ans_dir)

  data_len = np.shape(x_spec)[0]
  total_batch = data_len // NNET_PARAM.batch_size if data_len % NNET_PARAM.batch_size == 0 else (
      data_len // NNET_PARAM.batch_size)+1
  for i_batch in range(total_batch):
    s_time = time.time()
    cleaned, mask = sess.run([model.cleaned,model.mask])
    print('mask max min:', np.max(mask),np.min(mask))
    # print('mask max:',np.max(mask[0]),np.max(mask[1]),np.max(mask[2]),np.max(mask[3]))
    # print('mask min:',np.min(mask[0]),np.min(mask[1]),np.min(mask[2]),np.min(mask[3]))
    # print(np.sum(np.where(mask>1.1,1,0)),np.shape(mask)[0]*np.shape(mask)[1]*np.shape(mask)[2]) # percent >1.1
    # cleaned = sess.run(model.cleaned)
    s_site = i_batch*NNET_PARAM.batch_size
    e_site = min(s_site+NNET_PARAM.batch_size, data_len)
    for i in range(s_site, e_site):
      show_onewave(decode_ans_dir, str(i),
                   x_spec[i][:lengths[i]],
                   y_spec[i][:lengths[i]],
                   x_theta[i][:lengths[i]],
                   y_theta[i][:lengths[i]],
                   cleaned[i-s_site][:lengths[i]],
                   mixed_wave[i],
                   mask[i-s_site][:lengths[i]])
    e_time = time.time()-s_time
    print("One batch time: ",e_time)

  sess.close()
  tf.logging.info("Decoding done.")

def decode_oneset_new(setname, set_index_list_dir, ckpt_dir='nnet'):
  dataset_index_file = open(set_index_list_dir, 'r')
  dataset_index_strlist = dataset_index_file.readlines()
  if len(dataset_index_strlist) <= 0:
    print('Set %s have no element.' % setname)
    return

  decode_ans_dir = os.path.join(
      NNET_PARAM.SAVE_DIR, 'decode_ans', setname)
  if os.path.exists(decode_ans_dir):
    shutil.rmtree(decode_ans_dir)
  os.makedirs(decode_ans_dir)

  data_len = len(dataset_index_strlist)
  total_batch = data_len // NNET_PARAM.batch_size if data_len % NNET_PARAM.batch_size == 0 else (
      data_len // NNET_PARAM.batch_size)+1
  for i_batch in range(total_batch):
    s_time = time.time()
    # region preparation one batch data
    s_site = i_batch*NNET_PARAM.batch_size
    e_site = min(s_site+NNET_PARAM.batch_size, data_len)
    x_spec = []
    y_spec = []
    x_theta = []
    y_theta = []
    lengths = []
    mixed_wave = []
    for i, index_str in enumerate(dataset_index_strlist[s_site:e_site]):
      uttdir1, uttdir2 = index_str.replace('\n', '').split(' ')
      # print(uttdir1,uttdir2)
      uttwave1, uttwave2 = wav_tool._get_waveData1_waveData2_MAX_Volume(
          uttdir1, uttdir2)
      if uttdir2 != 'None':  # 将帧级语音和噪音混合后解码
        if MIXED_AISHELL_PARAM.MIX_METHOD == 'SNR':
          mixed_wave_t = wav_tool._mix_wav_by_SNR(uttwave1, uttwave2)
        if MIXED_AISHELL_PARAM.MIX_METHOD == 'LINEAR':
          mixed_wave_t = wav_tool._mix_wav_by_LINEAR(uttwave1, uttwave2)
        # mixed_wave_t = wav_tool._mix_wav_by_SNR(uttwave1, uttwave2)
      else:  # 解码单一混合语音（uttwave1是带有噪声的语音）
        mixed_wave_t = uttwave1
      x_spec_t = wav_tool._extract_norm_log_mag_spec(mixed_wave_t)
      y_spec_t = wav_tool._extract_norm_log_mag_spec(uttwave1)
      x_theta_t = wav_tool._extract_phase(mixed_wave_t)
      y_theta_t = wav_tool._extract_phase(uttwave1)
      mixed_wave.append(mixed_wave_t)
      x_spec.append(x_spec_t)
      y_spec.append(y_spec_t)
      x_theta.append(x_theta_t)
      y_theta.append(y_theta_t)
      lengths.append(np.shape(x_spec_t)[0])

    #  multi_single_mixed_wave_test
    max_length = np.max(lengths)
    x_spec = [np.pad(x_spec_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
              for i, x_spec_t in enumerate(x_spec)]
    y_spec = [np.pad(y_spec_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
              for i, y_spec_t in enumerate(y_spec)]
    x_theta = [np.pad(x_theta_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
               for i, x_theta_t in enumerate(x_theta)]
    y_theta = [np.pad(y_theta_t, ((0, max_length-lengths[i]), (0, 0)), 'constant', constant_values=0)
               for i, y_theta_t in enumerate(y_theta)]

    x_spec = np.array(x_spec, dtype=np.float32)
    y_spec = np.array(y_spec, dtype=np.float32)
    x_theta = np.array(x_theta, dtype=np.float32)
    y_theta = np.array(y_theta, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    # endregion

    # region load model
    g = tf.Graph()
    with g.as_default():
      with tf.device('/cpu:0'):
        with tf.name_scope('input'):
          dataset = tf.data.Dataset.from_tensor_slices(
              (x_spec, y_spec, x_theta, y_theta, lengths))
          dataset = dataset.batch(NNET_PARAM.batch_size)
          dataset_iter = dataset.make_one_shot_iterator()
          # dataset_iter = dataset.make_initializable_iterator()
          x_batch, y_batch, x_theta_batch, y_theta_batch, lengths_batch = dataset_iter.get_next()
      with tf.name_scope('model'):
        model = SE_MODEL(x_batch, y_batch, lengths_batch, x_theta_batch, y_theta_batch,
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
    # endregion
    cleaned, mask = sess.run([model.cleaned,model.mask])
    for i in range(e_site-s_site):
      show_onewave(decode_ans_dir, str(i+s_site),
                   x_spec[i][:lengths[i]],
                   y_spec[i][:lengths[i]],
                   x_theta[i][:lengths[i]],
                   y_theta[i][:lengths[i]],
                   cleaned[i][:lengths[i]],
                   mixed_wave[i],
                   mask[i][:lengths[i]])
    e_time = time.time()-s_time
    sess.close()
    tf.reset_default_graph()
    del dataset
    del dataset_iter
    del g
    del sess
    del x_spec
    del y_spec
    del x_theta
    del y_theta
    del lengths
    del mixed_wave
    gc.collect()
    # time.sleep(20)
    print("One batch time: ",e_time)

def decode_by_index():
  set_list = os.listdir('_decode_index')
  for list_file in set_list:
    if list_file[-4:] == 'list':
      # print(list_file)
      s_time=time.time()
      decode_oneset_old(
          list_file[:-5], os.path.join('_decode_index', list_file), ckpt_dir=NNET_PARAM.CHECK_POINT)
      e_time=time.time()-s_time
      print("cost Time: ",e_time)


def train_one_epoch(sess, tr_model):
  """Runs the model one epoch on given data."""
  tr_loss, i = 0, 0
  stime = time.time()
  while True:
    try:
      _, loss, current_batchsize = sess.run(
          # [tr_model.train_op, tr_model.loss, tf.shape(tr_model.lengths)[0]])
          [tr_model.train_op, tr_model.loss, tr_model.batch_size])
      tr_loss += loss
      if (i+1) % NNET_PARAM.minibatch_size == 0:
        lr = sess.run(tr_model.lr)
        costtime = time.time()-stime
        stime = time.time()
        print("MINIBATCH %05d: TRAIN AVG.LOSS %04.6f, "
              "(learning rate %02.6f)" % (
                  i + 1, tr_loss / (i*NNET_PARAM.batch_size+current_batchsize), lr), 'DURATION: %06dS' % costtime)
        sys.stdout.flush()
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= ((i-1)*NNET_PARAM.batch_size+current_batchsize)
  return tr_loss


def eval_one_epoch(sess, val_model):
  """Cross validate the model on given data."""
  val_loss = 0
  data_len = 0
  while True:
    try:
      loss, current_batchsize = sess.run(
          [val_model.loss, val_model.batch_size])
      # print(inputss)
      # exit(0)
      val_loss += loss
      data_len += current_batchsize
    except tf.errors.OutOfRangeError:
      break
  val_loss /= data_len
  return val_loss


def train():

  g = tf.Graph()
  with g.as_default():
    # region TFRecord+DataSet
    # tf.data with cpu is faster, but padded_batch may not surpport.
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        train_tfrecords, val_tfrecords, testcc_tfrecords = generate_tfrecord(
            gen=MIXED_AISHELL_PARAM.GENERATE_TFRECORD)
        if MIXED_AISHELL_PARAM.GENERATE_TFRECORD:
          print("TFRecords preparation over.")
          # exit(0)  # set gen=True and exit to generate tfrecords

        PSIRM = True if NNET_PARAM.MASK_TYPE == 'PSIRM' else False
        x_batch_tr, y_batch_tr, Xtheta_batch_tr, Ytheta_batch_tr, lengths_batch_tr, iter_train = get_batch_use_tfdata(
            train_tfrecords,
            get_theta=PSIRM)
        x_batch_val, y_batch_val,  Xtheta_batch_val, Ytheta_batch_val, lengths_batch_val, iter_val = get_batch_use_tfdata(
            val_tfrecords,
            get_theta=PSIRM)
    # endregion

    # build model
    with tf.name_scope('model'):
      tr_model = SE_MODEL(x_batch_tr,
                          y_batch_tr,
                          lengths_batch_tr,
                          Xtheta_batch_tr,
                          Ytheta_batch_tr)
      tf.get_variable_scope().reuse_variables()
      val_model = SE_MODEL(x_batch_val,
                           y_batch_val,
                           lengths_batch_val,
                           Xtheta_batch_val,
                           Ytheta_batch_val)

    utils.tf_tool.show_all_variables()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = NNET_PARAM.GPU_RAM_ALLOW_GROWTH
    config.allow_soft_placement = False
    sess = tf.Session(config=config)
    sess.run(init)

    # resume training
    if NNET_PARAM.resume_training.lower() == 'true':
      ckpt = tf.train.get_checkpoint_state(NNET_PARAM.SAVE_DIR + '/nnet')
      if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info("restore from" + ckpt.model_checkpoint_path)
        tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
        best_path = ckpt.model_checkpoint_path
      else:
        tf.logging.fatal("checkpoint not found")
      with open(os.path.join(NNET_PARAM.SAVE_DIR, 'train.log'), 'a+') as f:
        f.writelines('Training resumed.\n')
    else:
      if os.path.exists(os.path.join(NNET_PARAM.SAVE_DIR, 'train.log')):
        os.remove(os.path.join(NNET_PARAM.SAVE_DIR, 'train.log'))

    # validation before training.
    valstart_time = time.time()
    sess.run(iter_val.initializer)
    loss_prev = eval_one_epoch(sess,
                               val_model)
    tf.logging.info("CROSSVAL PRERUN AVG.LOSS %.4F  costime %dS" %
                    (loss_prev, time.time()-valstart_time))

    tr_model.assign_lr(sess, NNET_PARAM.learning_rate)
    g.finalize()

    # epochs training
    reject_num = 0
    for epoch in range(NNET_PARAM.start_epoch, NNET_PARAM.max_epochs):
      sess.run([iter_train.initializer, iter_val.initializer])
      start_time = time.time()

      # train one epoch
      tr_loss = train_one_epoch(sess,
                                tr_model)

      # Validation
      val_loss = eval_one_epoch(sess,
                                val_model)

      end_time = time.time()

      # Determine checkpoint path
      ckpt_name = "nnet_iter%d_lrate%e_trloss%.4f_cvloss%.4f_duration%ds" % (
          epoch + 1, NNET_PARAM.learning_rate, tr_loss, val_loss, end_time - start_time)
      ckpt_dir = NNET_PARAM.SAVE_DIR + '/nnet'
      if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
      ckpt_path = os.path.join(ckpt_dir, ckpt_name)

      # Relative loss between previous and current val_loss
      rel_impr = np.abs(loss_prev - val_loss) / loss_prev
      # Accept or reject new parameters
      msg = ""
      if val_loss < loss_prev:
        reject_num = 0
        tr_model.saver.save(sess, ckpt_path)
        # Logging train loss along with validation loss
        loss_prev = val_loss
        best_path = ckpt_path
        msg = ("Iteration %03d: TRAIN AVG.LOSS %.4f, lrate%e, VAL AVG.LOSS %.4f,\n"
               "%s, ckpt(%s) saved,\nEPOCH DURATION: %.2fs") % (
            epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
            "NNET Accepted", ckpt_name, end_time - start_time)
        tf.logging.info(msg)
      else:
        reject_num += 1
        tr_model.saver.restore(sess, best_path)
        msg = ("ITERATION %03d: TRAIN AVG.LOSS %.4f, (lrate%e) VAL AVG.LOSS %.4f,\n"
               "%s, ckpt(%s) abandoned,\nEPOCH DURATION: %.2fs") % (
            epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
            "NNET Rejected", ckpt_name, end_time - start_time)
        tf.logging.info(msg)
      with open(os.path.join(NNET_PARAM.SAVE_DIR, 'train.log'), 'a+') as f:
        f.writelines(msg+'\n')

      # Start halving when improvement is lower than start_halving_impr
      if (rel_impr < NNET_PARAM.start_halving_impr) or (reject_num >= 2):
        reject_num = 0
        NNET_PARAM.learning_rate *= NNET_PARAM.halving_factor
        tr_model.assign_lr(sess, NNET_PARAM.learning_rate)

      # Stopping criterion
      if rel_impr < NNET_PARAM.end_halving_impr:
        if epoch < NNET_PARAM.min_epochs:
          tf.logging.info(
              "we were supposed to finish, but we continue as "
              "min_epochs : %s" % NNET_PARAM.min_epochs)
          continue
        else:
          tf.logging.info(
              "finished, too small rel. improvement %g" % rel_impr)
          break

    sess.close()
    tf.logging.info("Done training")


def main(_):
  if not os.path.exists(NNET_PARAM.SAVE_DIR):
    os.makedirs(NNET_PARAM.SAVE_DIR)
  if NNET_PARAM.decode:
    if NNET_PARAM.decode == 1:
      decode_by_index()
    else:
      decode_testSet_get_SDR_Impr()
  else:
    train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)

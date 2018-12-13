import time
import tensorflow as tf
import numpy as np
import sys
import utils
import os
import shutil
from models.lstm_SE import SE_MODEL
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM
from dataManager.mixed_aishell_tfrecord_io import generate_tfrecord, get_batch_use_tfdata


def decode():
  pass

def train_one_epoch(sess, tr_model, i_epoch, run_metadata):
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


def eval_one_epoch(sess, val_model, run_metadata):
  """Cross validate the model on given data."""
  val_loss = 0
  data_len = 0
  while True:
    try:
      loss, current_batchsize = sess.run(
          [val_model.loss, val_model.batch_size])
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
          exit(0)  # set gen=True and exit to generate tfrecords
        x_batch_tr, y1_batch_tr, y2_batch_tr, Xtheta_batch_tr, Ytheta_batch_tr, lengths_batch_tr, iter_train = get_batch_use_tfdata(
            train_tfrecords,
            get_theta=False)
        x_batch_val, y1_batch_val, y2_batch_val, Xtheta_batch_val, Ytheta_batch_val, lengths_batch_val, iter_val = get_batch_use_tfdata(
            val_tfrecords,
            get_theta=False)
    # endregion

    # build model
    with tf.name_scope('model'):
      tr_model = SE_MODEL(x_batch_tr,
                          y1_batch_tr,
                          y2_batch_tr,
                          lengths_batch_tr,
                          Xtheta_batch_tr,
                          Ytheta_batch_tr)
      tf.get_variable_scope().reuse_variables()
      val_model = SE_MODEL(x_batch_val,
                           y1_batch_val,
                           y2_batch_val,
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

    # prepare run_metadata for timeline
    run_metadata = None
    if NNET_PARAM.time_line:
      run_metadata = tf.RunMetadata()
      if os.path.exists('_timeline'):
        shutil.rmtree('_timeline')
      os.mkdir('_timeline')

    # validation before training.
    valstart_time = time.time()
    sess.run(iter_val.initializer)
    loss_prev = eval_one_epoch(sess,
                               val_model,
                               run_metadata)
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
                                tr_model,
                                epoch,
                                run_metadata)

      # Validation
      val_loss = eval_one_epoch(sess,
                                val_model,
                                run_metadata)

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
      with open(os.path.join(NNET_PARAM.save_dir, 'train.log'), 'a+') as f:
        f.writelines(msg+'\n')

      # Start halving when improvement is lower than start_halving_impr
      if (rel_impr < NNET_PARAM.start_halving_impr) or (reject_num >= 3):
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
    decode()
  else:
    train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)

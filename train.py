#coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import  absolute_import

import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
slim = tf.contrib.slim
import deeplab_model
import input_data
import utils.utils as Utils

flags = tf.app.flags
FLAGS = flags.FLAGS


# For dataset
BATCH_SIZE_OS_16 = 8
BATCH_SIZE_OS_8 = 2
CROP_HEIGHT = input_data.HEIGHT
CROP_WIDTH = input_data.WIDTH
CHANNELS = 3
CLASSES = deeplab_model.CLASSES
_IGNORE_LABEL = input_data._IGNORE_LABEL

PRETRAINED_MODEL_PATH = deeplab_model.PRETRAINED_MODEL_PATH

# For training steps
SAMPLES_AUG = 10582
SAMEPLES_train = 1464
EPOCHES = 42
MAX_STEPS_FAST = (SAMPLES_AUG) // BATCH_SIZE_OS_16 * EPOCHES
MAX_STEPS_SLOW = (SAMEPLES_train) // BATCH_SIZE_OS_8 * EPOCHES

SAVE_CHECKPOINT_STEPS = 5000
SAVE_SUMMARY_STEPS = 1000
PRINT_STEPS = 200

# For training config
_POWER = 0.9
_WEIGHT_DECAY = 1e-4

initial_lr_fast = 7e-3
end_lr_fast = 1e-5
decay_steps_fast = 50000


initial_lr_slow = 1e-5
end_lr_slow = 1e-6
decay_steps_slow = 30000


flags.DEFINE_integer('output_stride', 16, 'output stride used in the resnet model')

if FLAGS.output_stride == 16:
    MAX_STEPS = MAX_STEPS_FAST
    initial_lr = initial_lr_fast
    end_lr = end_lr_fast
    decay_steps = decay_steps_fast
    BATCH_SIZE = BATCH_SIZE_OS_16
    train_data = input_data.read_train_data()
    val_data = input_data.read_val_data()
elif FLAGS.output_stride == 8:
    MAX_STEPS = MAX_STEPS_SLOW
    initial_lr = initial_lr_slow
    end_lr = end_lr_slow
    decay_steps = decay_steps_slow
    BATCH_SIZE = BATCH_SIZE_OS_8
    train_data = input_data.read_train_raw_data()
    val_data = input_data.read_val_data()


# for saved path
saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './summary/train/'
saved_summary_test_path = './summary/test/'


def cal_loss(logits, y, loss_weight=1.0):
    '''
    raw_prediction = tf.reshape(logits, [-1, CLASSES])
    raw_gt = tf.reshape(y, [-1])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, CLASSES - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    '''

    y = tf.reshape(y, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(y, _IGNORE_LABEL)) * loss_weight
    one_hot_labels = tf.one_hot(
        y, CLASSES, on_value=1.0, off_value=0.0)
    logits = tf.reshape(logits, shape=[-1, CLASSES])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits, weights=not_ignore_mask)

    return tf.reduce_mean(loss)

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, CROP_HEIGHT, CROP_WIDTH, CHANNELS], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[None, CROP_HEIGHT, CROP_WIDTH], name='ground_truth')


logits = deeplab_model.deeplab_v3_plus(x, is_training=True, output_stride=FLAGS.output_stride, pre_trained_model=PRETRAINED_MODEL_PATH)

#logits = deeplab_model.deeplabv3_plus_model_fn(x)

with tf.name_scope('regularization'):

    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]
    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        l2_loss = _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])

with tf.name_scope('loss'):

    loss = cal_loss(logits, y)
    tf.summary.scalar('loss', loss)
    loss_all = loss + l2_loss
    tf.summary.scalar('loss_all', loss_all)


with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.polynomial_decay(
        learning_rate=initial_lr,
        global_step=global_step,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=_POWER,
        cycle=False,
        name=None
    )
    tf.summary.scalar('learning_rate', lr)



with tf.name_scope("opt"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_all, var_list=train_var_list, global_step=global_step)


with tf.name_scope("mIoU"):

    predictions = tf.argmax(logits, axis=-1, name='predictions')


    train_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('train_mIoU', train_mIoU)
    test_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('test_mIoU',test_mIoU)

merged = tf.summary.merge_all()


with tf.Session() as sess:


    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

        sess.run(tf.assign(global_step, 0))

        print("Model restored...")

    # saver.restore(sess, './checkpoint/deeplabv3plus.model-30000')

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)

    for i in range(0, MAX_STEPS + 1):



        image_batch_0, image_batch, anno_batch, filename = train_data.next_batch(BATCH_SIZE, is_training=True)
        image_batch_val_0, image_batch_val, anno_batch_val, filename_val = val_data.next_batch(BATCH_SIZE, is_training=True)


        _ = sess.run(optimizer, feed_dict={x: image_batch, y: anno_batch})


        if i % SAVE_SUMMARY_STEPS == 0:
            train_summary = sess.run(merged, feed_dict={x: image_batch, y: anno_batch})
            train_summary_writer.add_summary(train_summary, i)
            test_summary = sess.run(merged, feed_dict={x: image_batch_val, y: anno_batch_val})
            test_summary_writer.add_summary(test_summary, i)


        if i % PRINT_STEPS == 0:
            train_loss_val_all = sess.run(loss_all, feed_dict={x: image_batch, y: anno_batch})
            print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d, | Train loss all: %f" % (i, train_loss_val_all))

        if i % SAVE_SUMMARY_STEPS == 0:
            learning_rate = sess.run(lr)
            pred_train, train_loss_val_all, train_loss_val = sess.run([predictions, loss_all, loss],
                                                                      feed_dict={x: image_batch, y: anno_batch})
            pred_test, test_loss_val_all, test_loss_val = sess.run([predictions, loss_all, loss],
                                                                   feed_dict={x: image_batch_val, y: anno_batch_val})

            train_mIoU_val, train_IoU_val = Utils.cal_batch_mIoU(pred_train, anno_batch, CLASSES)
            test_mIoU_val, test_IoU_val = Utils.cal_batch_mIoU(pred_test, anno_batch_val, CLASSES)

            sess.run(tf.assign(train_mIoU, train_mIoU_val))
            sess.run(tf.assign(test_mIoU, test_mIoU_val))
            print('------------------------------')
            print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d, | Lr: %f, | train loss all: %f, | train loss: %f, | train mIoU: %f, | test loss all: %f, | test loss: %f, | test mIoU: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val, test_loss_val_all, test_loss_val, test_mIoU_val))
            print('------------------------------')
            print(train_IoU_val)
            print(test_IoU_val)
            print('------------------------------')
            #prediction = tf.argmax(logits, axis=-1, name='predictions')

        if i % SAVE_CHECKPOINT_STEPS == 0:
            if not os.path.exists('images'):
                os.mkdir('images')
            for j in range(BATCH_SIZE):
                cv2.imwrite('images/%d_%s_train_img.png' %(i, filename[j].split('.')[0]), image_batch[j])
                cv2.imwrite('images/%d_%s_train_anno.png' %(i, filename[j].split('.')[0]), Utils.color_gray(anno_batch[j]))
                cv2.imwrite('images/%d_%s_train_pred.png' %(i, filename[j].split('.')[0]), Utils.color_gray(pred_train[j]))
                cv2.imwrite('images/%d_%s_test_img.png' %(i, filename_val[j].split('.')[0]), image_batch_val[j])
                cv2.imwrite('images/%d_%s_test_anno.png' %(i, filename_val[j].split('.')[0]), Utils.color_gray(anno_batch_val[j]))
                cv2.imwrite('images/%d_%s_test_pred.png' %(i, filename_val[j].split('.')[0]), Utils.color_gray(pred_test[j]))


        if i % SAVE_CHECKPOINT_STEPS == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'deeplabv3plus.model'), global_step=i)

if __name__ == '__main__':
    tf.app.run()
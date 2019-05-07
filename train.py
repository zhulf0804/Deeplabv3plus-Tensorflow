#coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import  absolute_import

import tensorflow as tf
import numpy as np
import os
import cv2
slim = tf.contrib.slim
import deeplab_model
import input_data
import utils.utils as Utils


BATCH_SIZE = 4
CROP_HEIGHT = input_data.HEIGHT
CROP_WIDTH = input_data.WIDTH
CLASSES = deeplab_model.CLASSES
CHANNELS = 3

PRETRAINED_MODEL_PATH = deeplab_model.PRETRAINED_MODEL_PATH

SAMPLES = 10582
EPOCHES = 30
MAX_STEPS = (SAMPLES) // 4 * EPOCHES


initial_lr = 7e-3
_WEIGHT_DECAY = 1e-4

saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './summary/train/'
saved_summary_test_path = './summary/test/'

def weighted_loss(logits, labels, num_classes, head=None, ignore=19):
    """re-weighting"""
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        label_flat = tf.reshape(labels, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def cal_loss(logits, labels):

    loss_weight = [1.0 for i in range(CLASSES)]
    loss_weight = np.array(loss_weight)

    labels = tf.cast(labels, tf.int32)

    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=CLASSES, head=loss_weight)

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, CROP_HEIGHT, CROP_WIDTH, CHANNELS], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, CROP_HEIGHT, CROP_WIDTH], name='ground_truth')


logits = deeplab_model.deeplab_v3_plus(x, is_training=True, output_stride=16, pre_trained_model=PRETRAINED_MODEL_PATH)

with tf.name_scope('regularization'):

    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]
    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        l2_loss = _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])

with tf.name_scope('loss'):
    #reshaped_logits = tf.reshape(logits, [BATCH_SIZE, -1])
    #reshape_y = tf.reshape(y, [BATCH_SIZE, -1])
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshape_y, logits=reshaped_logits), name='loss')
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
    loss = cal_loss(logits, y)
    tf.summary.scalar('loss', loss)
    loss_all = loss + l2_loss
    #loss_all = loss
    tf.summary.scalar('loss_all', loss_all)

with tf.name_scope('learning_rate'):
    lr = tf.Variable(initial_lr, dtype=tf.float32)
    tf.summary.scalar('learning_rate', lr)

optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_all)

with tf.name_scope("mIoU"):
    softmax = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(logits, axis=-1, name='predictions')

    train_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('train_mIoU', train_mIoU)
    test_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('test_mIoU',test_mIoU)

merged = tf.summary.merge_all()

train_data = input_data.read_train_data()
test_data = input_data.read_test_data()

with tf.Session() as sess:


    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # saver.restore(sess, './checkpoint/deeplabv3plus.model-30000')

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)

    for i in range(0, MAX_STEPS + 1):


        image_batch_0, image_batch, anno_batch, filename = train_data.next_batch(BATCH_SIZE, 'train')
        image_batch_test_0, image_batch_test, anno_batch_test, filename_test = test_data.next_batch(BATCH_SIZE)

        _ = sess.run(optimizer, feed_dict={x: image_batch, y: anno_batch})

        train_summary = sess.run(merged, feed_dict={x: image_batch, y: anno_batch})
        train_summary_writer.add_summary(train_summary, i)
        test_summary = sess.run(merged, feed_dict={x: image_batch_test, y: anno_batch_test})
        test_summary_writer.add_summary(test_summary, i)

        pred_train, train_loss_val_all, train_loss_val = sess.run([predictions, loss_all, loss], feed_dict={x: image_batch, y: anno_batch})
        pred_test, test_loss_val_all, test_loss_val = sess.run([predictions, loss_all, loss], feed_dict={x: image_batch_test, y: anno_batch_test})

        learning_rate = sess.run(lr)

        if i % 10 == 0:
            print(
                "train step: %d, train loss all: %f, train loss: %f, test loss all: %f, test loss: %f," % (
                    i, train_loss_val_all, train_loss_val, test_loss_val_all, test_loss_val))

        if i % 200 == 0:

            train_mIoU_val, train_IoU_val = Utils.cal_batch_mIoU(pred_train, anno_batch, CLASSES)
            test_mIoU_val, test_IoU_val = Utils.cal_batch_mIoU(pred_test, anno_batch_test, CLASSES)

            sess.run(tf.assign(train_mIoU, train_mIoU_val))
            sess.run(tf.assign(test_mIoU, test_mIoU_val))

            print(
                "train step: %d, learning rate: %f, train loss all: %f, train loss: %f, train mIoU: %f, test loss all: %f, test loss: %f, test mIoU: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val, test_loss_val_all, test_loss_val, test_mIoU_val))
            print(train_IoU_val)
            print(test_IoU_val)
            #prediction = tf.argmax(logits, axis=-1, name='predictions')

        if i % 1000 == 0:
            if not os.path.exists('images'):
                os.mkdir('images')
            for j in range(BATCH_SIZE):
                cv2.imwrite('images/train_img_%d_%s' %(j, filename[j]), image_batch[j])
                cv2.imwrite('images/train_anno_%d_%s' %(j, filename[j]), Utils.color_gray(anno_batch[j]))
                cv2.imwrite('images/train_pred_%d_%s' %(j, filename[j]), Utils.color_gray(pred_train[j]))
                cv2.imwrite('images/test_img_%d_%s' %(j, filename[j]), image_batch_test[j])
                cv2.imwrite('images/test_anno_%d_%s' %(j, filename[j]), Utils.color_gray(anno_batch_test[j]))
                cv2.imwrite('images/test_pred_%d_%s' %(j, filename[j]), Utils.color_gray(pred_test[j]))


        if i % 5000 == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'deeplabv3plus.model'), global_step=i)


        if i == 10000 or i == 40000 or i == 100000:
            sess.run(tf.assign(lr, 0.1 * lr))



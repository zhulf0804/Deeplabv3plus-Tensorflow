# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import deeplab_model
import input_data
import utils.utils as Utils


PRETRAINED_MODEL_PATH = deeplab_model.PRETRAINED_MODEL_PATH


BATCH_SIZE = 1
CLASSES = deeplab_model.CLASSES
saved_ckpt_path = './checkpoint/'
saved_prediction = './pred/'
prediction_on = 'val' # 'train', 'val' or 'test'


test_data = input_data.read_test_data()

with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 3], name='x_input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], name='ground_truth')

logits = deeplab_model.deeplab_v3_plus(x, is_training=True, output_stride=16, pre_trained_model=PRETRAINED_MODEL_PATH)


with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #saver.restore(sess, './checkpoint/deeplabv3plus.model-2000')

    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    print("predicting on %s set..." % prediction_on)


    for i in range(2):
        b_image_0, b_image, b_anno, b_filename =  test_data.next_batch(BATCH_SIZE, type=prediction_on, inference=True)

        pred = sess.run(prediction, feed_dict={x: b_image, y: b_anno})

        print(pred.shape, b_anno.shape)

        mIoU_val, IoU_val = Utils.cal_batch_mIoU(pred, b_anno, CLASSES)
        # save raw image, annotation, and prediction
        pred = pred.astype(np.uint8)
        b_anno = b_anno.astype(np.uint8)
        pred_color = Utils.color_gray(pred[0, :, :])
        b_anno_color = Utils.color_gray(b_anno[0, :, :])

        b_image_0 = b_image_0.astype(np.uint8)

        img = Image.fromarray(b_image_0[0])
        anno = Image.fromarray(b_anno_color)
        pred = Image.fromarray(pred_color)

        basename = b_filename.split('.')[0]
        #print(basename)

        if not os.path.exists(saved_prediction):
            os.mkdir(saved_prediction)
        img.save(os.path.join(saved_prediction, basename + '.png'))
        anno.save(os.path.join(saved_prediction, basename + '_anno.png'))
        pred.save(os.path.join(saved_prediction, basename + '_pred.png'))

        print("%s.png: prediction saved in %s, mIoU value is %.2f" % (basename, saved_prediction, mIoU_val))
        print(IoU_val)



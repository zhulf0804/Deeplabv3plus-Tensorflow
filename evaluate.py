# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
import datetime
import matplotlib.pyplot as plt

import deeplab_model
import input_data
import utils.utils as Utils


PRETRAINED_MODEL_PATH = deeplab_model.PRETRAINED_MODEL_PATH


BATCH_SIZE = 1
CLASSES = deeplab_model.CLASSES
saved_ckpt_path = './checkpoint/'
saved_prediction_val_color = './pred/val_color'
saved_prediction_val_gray = './pred/val_gray'

saved_prediction_test_color = './pred/test_color'
saved_prediction_test_gray = './pred/test_gray'

VAL_LIST = input_data.VAL_LIST
ANNOTATION_PATH = input_data.ANNOTATION_PATH


val_num = 1449
test_num = 1456


if not os.path.exists('./pred'):
    os.mkdir('./pred')

val_data = input_data.read_val_data()
test_data = input_data.read_test_data()

with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 3], name='x_input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], name='ground_truth')

logits = deeplab_model.deeplab_v3_plus(x, is_training=False, output_stride=16, pre_trained_model=PRETRAINED_MODEL_PATH)



with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')


def get_val_predictions():
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # saver.restore(sess, './checkpoint/deeplabv3plus.model-5000')

        ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        print("predicting on val set...")

        for i in range(val_num):
            b_image_0, b_image, b_anno, b_filename = val_data.next_batch(BATCH_SIZE, is_training=False, Shuffle=False)

            pred = sess.run(prediction, feed_dict={x: b_image})

            basename = b_filename.split('.')[0]

            if i % 100 == 0:
                print(i, pred.shape)
                print(basename)

            # save raw image, annotation, and prediction
            pred = pred.astype(np.uint8)
            b_anno = b_anno.astype(np.uint8)

            pred_color = Utils.color_gray(pred[0, :, :])
            b_anno_color = Utils.color_gray(b_anno[0, :, :])

            b_image_0 = b_image_0.astype(np.uint8)

            pred_gray = Image.fromarray(pred[0])
            img = Image.fromarray(b_image_0[0])
            anno = Image.fromarray(b_anno_color)
            pred = Image.fromarray(pred_color)

            if not os.path.exists(saved_prediction_val_gray):
                os.mkdir(saved_prediction_val_gray)
            pred_gray.save(os.path.join(saved_prediction_val_gray, basename + '.png'))

            if not os.path.exists(saved_prediction_val_color):
                os.mkdir(saved_prediction_val_color)
            img.save(os.path.join(saved_prediction_val_color, basename + '_raw.png'))
            anno.save(os.path.join(saved_prediction_val_color, basename + '_anno.png'))
            pred.save(os.path.join(saved_prediction_val_color, basename + '_pred.png'))

    print("predicting on val set finished")


def get_test_predictions():

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # saver.restore(sess, './checkpoint/deeplabv3plus.model-5000')

        ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        print("predicting on test set...")

        for i in range(test_num):
            b_image_0, b_image, b_anno, b_filename = test_data.next_batch(BATCH_SIZE, is_training=False, Shuffle=False)

            pred = sess.run(prediction, feed_dict={x: b_image})

            basename = b_filename.split('.')[0]


            if i % 100 == 0:
                print(i, pred.shape)
                print(basename)

            # save raw image, annotation, and prediction
            pred = pred.astype(np.uint8)
            pred_color = Utils.color_gray(pred[0, :, :])
            b_anno_color = Utils.color_gray(b_anno[0, :, :])

            b_image_0 = b_image_0.astype(np.uint8)

            img = Image.fromarray(b_image_0[0])
            pred = Image.fromarray(pred_color)
            pred_gray = Image.fromarray(pred[0, :, :])



            if not os.path.exists(saved_prediction_test_gray):
                os.mkdir(saved_prediction_test_gray)
            pred_gray.save(os.path.join(saved_prediction_test_gray, basename + '.png'))

            if not os.path.exists(saved_prediction_test_color):
                os.mkdir(saved_prediction_test_color)
            img.save(os.path.join(saved_prediction_test_color, basename + '_raw.png'))
            pred.save(os.path.join(saved_prediction_val_color, basename + '_pred.png'))

    print("predicting on test set finished")

def get_val_mIoU():

    print("Start to get mIoU on val set...")

    f = open(VAL_LIST)
    lines = f.readlines()
    annotation_files = [os.path.join(ANNOTATION_PATH, line.strip() + '.png') for line in lines]
    prediction_files = [os.path.join(saved_prediction_val_gray, line.strip() + '.png') for line in lines]


    for i, annotation_file in enumerate(annotation_files):

        annotation_data = cv2.imread(annotation_file, cv2.IMREAD_GRAYSCALE)
        annotation_data = annotation_data.reshape(-1)
        if i == 0:
            annotations_data = annotation_data
        else:
            annotations_data = np.concatenate((annotations_data, annotation_data))

    print(annotations_data.shape)
    for i, prediction_file in enumerate(prediction_files):
        prediction_data = cv2.imread(prediction_file, cv2.IMREAD_GRAYSCALE)
        prediction_data = prediction_data.reshape(-1)
        if i == 0:
            predictions_data = prediction_data
        else:

            predictions_data = np.concatenate((predictions_data, prediction_data))

    print(predictions_data.shape)

    mIoU, IoU = Utils.cal_batch_mIoU(predictions_data, annotations_data, CLASSES)

    print(mIoU)
    print(IoU)

if __name__ == '__main__':
    get_val_predictions()
    get_val_mIoU()




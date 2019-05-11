# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import deeplab_model
import deeplab_model_0
import input_data
import utils.utils as Utils

IMAGE_PATH = input_data.IMAGE_PATH
ANNOTATION_PATH = input_data.ANNOTATION_PATH

PRETRAINED_MODEL_PATH = deeplab_model.PRETRAINED_MODEL_PATH


BATCH_SIZE = 1
CLASSES = deeplab_model.CLASSES
saved_ckpt_path = './checkpoint/'
saved_prediction = './pred/'

predictions_on = ['train', 'val', 'test']

prediction_on = predictions_on[1]  # select the dataset to predict

if prediction_on == 'train':
    data = input_data.read_train_data()
elif prediction_on == 'val':
    data = input_data.read_val_data()
elif prediction_on == 'test':
    data = input_data.read_test_data()



with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 3], name='x_input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], name='ground_truth')

#logits = deeplab_model_0.deeplab_v3_plus(x, is_training=True, output_stride=16, pre_trained_model=PRETRAINED_MODEL_PATH)
logits = deeplab_model.deeplabv3_plus_model_fn(x, is_training=False)


with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_on_which', type=str, default='', help='which dataset to predict')
    parser.add_argument('--filename', type=str, default='', help='the filename')

    args = parser.parse_args()

    filename = args.filename
    prediction_on_which = args.prediction_on_which

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # saver.restore(sess, './checkpoint/deeplabv3plus.model-5000')

        ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        if filename == '' or prediction_on_which == '':
            b_image_0, b_image, b_anno, b_filename = data.next_batch(BATCH_SIZE, is_training=False, Shuffle=False)
        else:
            filename_path = os.path.join(IMAGE_PATH, filename + '.jpg')
            if not os.path.exists(filename_path):
                raise Exception('%s not exist' %filename_path)

            b_image_0 = cv2.imread(filename_path)
            b_image = np.expand_dims(input_data.mean_substraction(b_image_0), 0)
            print(b_image)
            b_anno = np.expand_dims(cv2.imread(os.path.join(ANNOTATION_PATH, filename + '.png'), cv2.IMREAD_GRAYSCALE), 0)
            b_filename = filename

            prediction_on = prediction_on_which


        print("predicting on %s set..." % prediction_on)


        pred = sess.run(prediction, feed_dict={x: b_image})

        print(pred.shape, b_anno.shape)

        # save raw image, annotation, and prediction
        pred = pred.astype(np.uint8)
        pred_color = Utils.color_gray(pred[0, :, :])

        b_image_0 = b_image_0.astype(np.uint8)

        basename = b_filename.split('.')[0]

        if not os.path.exists(saved_prediction):
            os.mkdir(saved_prediction)

        if prediction_on == 'train' or prediction_on == 'val':

            b_anno = b_anno.astype(np.uint8)
            b_anno_color = Utils.color_gray(b_anno[0, :, :])
            anno = Image.fromarray(b_anno_color)
            anno.save(os.path.join(saved_prediction, basename + '_anno.png'))

            mIoU_val, IoU_val = Utils.cal_batch_mIoU(pred, b_anno, CLASSES)
            print("%s.png: prediction saved in %s, mIoU value is %.2f" % (basename, saved_prediction, mIoU_val))
            print(IoU_val)
        else:

            print("%s.png: prediction saved in %s" % (basename, saved_prediction))

        img = Image.fromarray(b_image_0[0])

        pred = Image.fromarray(pred_color)

        img.save(os.path.join(saved_prediction, basename + '_raw.png'))
        pred.save(os.path.join(saved_prediction, basename + '_pred.png'))


## Exec
## python predict.py
## or
## python predict.py --prediction_on_which val --filename 2009_003804


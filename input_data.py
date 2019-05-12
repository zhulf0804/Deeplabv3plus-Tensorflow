# coding=utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
from sklearn.utils import shuffle
# import tensorflow as tf
import cv2
import random
import math

HEIGHT = 512
WIDTH = 512
CHANNELS = 3

CLASSES = 21

_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_R_MEAN = 123.15
_G_MEAN = 115.90
_B_MEAN = 103.06
_MEAN_RGB = [_R_MEAN, _G_MEAN, _B_MEAN]

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]



dataset = '/Volumes/Samsung_T5/datasets/VOCdevkit' # Select your path

tfrecord_file = os.path.join(dataset, 'tfrecord')
TRAIN_LIST = os.path.join(dataset, 'train.txt')
VAL_LIST = os.path.join(dataset, 'val.txt')
TEST_LIST = os.path.join(dataset, 'test.txt')
IMAGE_PATH = os.path.join(dataset, 'VOC2012/JPEGImages')
ANNOTATION_PATH = os.path.join(dataset, 'VOC2012/SegmentationClassAug')



def flip_random_left_right(image, anno):
    '''
    :param image: [height, width, channel]
    :return:
    '''
    flag = random.randint(0, 1)
    if flag:
        return cv2.flip(image, 1), cv2.flip(anno, 1)
    return image, anno


def random_pad_crop(image, anno):

    image = image.astype(np.float32)

    height, width = anno.shape

    #padded_image = np.pad(image, ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width), (0, 0)), mode='constant', constant_values=_MEAN_RGB)

    padded_image_r = np.pad(image[:, :, 0], ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width)), mode='constant', constant_values=_R_MEAN)
    padded_image_g = np.pad(image[:, :, 1], ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width)), mode='constant', constant_values=_G_MEAN)
    padded_image_b = np.pad(image[:, :, 2], ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width)), mode='constant', constant_values=_B_MEAN)
    padded_image = np.zeros(shape=[np.maximum(height, HEIGHT), np.maximum(width, WIDTH), 3], dtype=np.float32)
    padded_image[:, :, 0] = padded_image_r
    padded_image[:, :, 1] = padded_image_g
    padded_image[:, :, 2] = padded_image_b

    padded_anno = np.pad(anno, ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width)), mode='constant', constant_values=_IGNORE_LABEL)

    y = random.randint(0, np.maximum(height, HEIGHT) - HEIGHT)
    x = random.randint(0, np.maximum(width, WIDTH) - WIDTH)

    cropped_image = padded_image[y:y+HEIGHT, x:x+WIDTH, :]
    cropped_anno = padded_anno[y:y+HEIGHT, x:x+WIDTH]

    return cropped_image, cropped_anno


def random_resize(image, anno):
    height, width = anno.shape

    scale = random.uniform(_MIN_SCALE, _MAX_SCALE)
    scale_image = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)
    scale_anno = cv2.resize(anno, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)

    return scale_image, scale_anno


def mean_substraction(image):
    substraction_mean_image = np.zeros_like(image, dtype=np.float32)
    substraction_mean_image[:, :, 0] = image[:, :, 0] - _R_MEAN
    substraction_mean_image[:, :, 1] = image[:, :, 1] - _G_MEAN
    substraction_mean_image[:, :, 2] = image[:, :, 2] - _B_MEAN

    return substraction_mean_image


def augment(img, anno):

    scale_img, scale_anno = random_resize(img, anno)

    img = img.astype(np.float32)
    cropped_image, cropped_anno = random_pad_crop(scale_img, scale_anno)


    flipped_img, flipped_anno = flip_random_left_right(cropped_image, cropped_anno)

    substracted_img = mean_substraction(flipped_img)

    return substracted_img, flipped_anno


class Dataset(object):

    def __init__(self, img_filenames, anno_filenames):
        self._num_examples = len(anno_filenames)
        self._image_data = img_filenames
        self._labels = anno_filenames
        self._epochs_done = 0
        self._index_in_epoch = 0
        self._flag = 0

    def next_batch(self, batch_size, is_training=False, Shuffle=True):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

            if Shuffle:
                self._image_data, self._labels = shuffle(self._image_data, self._labels)

        end = self._index_in_epoch

        batch_img_raw = np.zeros([batch_size, HEIGHT, WIDTH, 3], dtype=np.float32)
        batch_img = np.zeros([batch_size, HEIGHT, WIDTH, 3], dtype=np.float32)
        batch_anno = np.zeros([batch_size, HEIGHT, WIDTH], dtype=np.uint8)
        filenames = []
        for i in range(start, end):
            img = cv2.imread(self._image_data[i])
            img = img[:,:,::-1]
            anno = cv2.imread(self._labels[i], cv2.IMREAD_GRAYSCALE)

            if is_training:
                aug_img, aug_anno = augment(img, anno)

                height, width, _ = img.shape
                batch_img_raw[i-start, 0:np.minimum(height, HEIGHT), 0:np.minimum(width, WIDTH), :] = img[0:np.minimum(height, HEIGHT), 0:np.minimum(width, WIDTH), :]
                batch_img[i-start, ...] = aug_img
                batch_anno[i-start, ...] = aug_anno
                filenames.append(os.path.basename(self._image_data[i]))

        if is_training:
            return batch_img_raw, batch_img, batch_anno, filenames
        else:
            inference_image = mean_substraction(img)
            print(os.path.basename(self._image_data[start]))
            return np.expand_dims(img, 0), np.expand_dims(inference_image, 0), np.expand_dims(anno, 0), os.path.basename(self._image_data[start])


def read_train_data(Shuffle=True):

    f = open(TRAIN_LIST)
    lines = f.readlines()
    img_filenames = [os.path.join(IMAGE_PATH, line.strip() + '.jpg') for line in lines]
    anno_filenames = [os.path.join(ANNOTATION_PATH, line.strip() + '.png') for line in lines]

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    train_data = Dataset(img_filenames, anno_filenames)

    return train_data


def read_val_data(Shuffle=True):
    f = open(VAL_LIST)
    lines = f.readlines()
    img_filenames = [os.path.join(IMAGE_PATH, line.strip() + '.jpg') for line in lines]
    anno_filenames = [os.path.join(ANNOTATION_PATH, line.strip() + '.png') for line in lines]

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    val_data = Dataset(img_filenames, anno_filenames)

    return val_data

def read_test_data(Shuffle=True):
    f = open(TEST_LIST)
    lines = f.readlines()
    img_filenames = [os.path.join(IMAGE_PATH, line.strip() + '.jpg') for line in lines]
    anno_filenames = [os.path.join(ANNOTATION_PATH, line.strip() + '.png') for line in lines]

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    test_data = Dataset(img_filenames, anno_filenames)

    return test_data


if __name__ == '__main__':
    train_data = read_train_data()
    test_data = read_val_data()
    train_img_raw, train_img_data, train_lables, train_filenames = train_data.next_batch(4, True)
    test_img_raw, test_img_data, test_labels, test_filenames = test_data.next_batch(1)

    #print(train_img_data)
    #print(test_img_data)

    for i in range(4):
        cv2.imwrite('test/trainraw_%d.png' % i, train_img_raw[i])
        cv2.imwrite('test/train_%d.png'%i, train_img_data[i])
        cv2.imwrite('test/train_labels_%d.png'%i, train_lables[i])
        print(train_filenames[i])

    print("===============")

    for i in range(1):
        cv2.imwrite('test/testraw_%d.png' % i, test_img_raw[i])

        cv2.imwrite('test/test_%d.png' % i, test_img_data[i])
        cv2.imwrite('test/test_labels_%d.png' % i, test_labels[i])
        print(test_filenames)

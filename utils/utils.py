# coding=utf-8
from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
import input_data
import sys
sys.path.append("..")

CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', \
                'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def cal_batch_mIoU(pred, gt, classes_num):
    """

    :param pred: [batch, height, width]
    :param gt: [batch, height, width]
    :param classes_num:
    :return:
    """
    IoU_0 = []
    IoU = []
    eps = 1e-6
    for i in range(classes_num):
        a = np.sum(pred == i)
        b = np.sum(gt == i)
        c = [pred == i, gt == i]
        c = np.sum(np.all(c, 0))
        iou = c / (a + b - c + eps)
        if b != 0:
            IoU.append(iou)
        IoU_0.append(round(iou, 2))

    IoU_0 = dict(zip(CLASS_NAMES, IoU_0))
    mIoU = np.mean(IoU)
    return mIoU, IoU_0



def color_gray(image):
    cmap = input_data.label_colours
    height, width = image.shape

    return_img = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                return_img[i, j, :] = (255, 255, 255)
            else:
                return_img[i, j, :] = cmap[image[i, j]]

    return return_img

if __name__ == '__main__':
    pass
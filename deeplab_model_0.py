#coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import math

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

CLASSES = 21

_BATCH_NORM_DECAY = 0.9997
_BATCH_NORM_EPSILON = 1e-5


PRETRAINED_MODEL_PATH = './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt'

def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)

    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(inputs, training):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def aspp(inputs, output_stride, is_training, depth=256):
    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2 * rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):

        input_shape = inputs.get_shape().as_list()
        #input_size = input_shape[1:3]
        input_size = tf.shape(inputs)[1:3]
        in_channels = input_shape[-1]

        # (a)
        with tf.name_scope('aspp_a'):
            weight_1 = weight_variable([1, 1, in_channels, depth], name='weight_1x1')
            weight_3_1 = weight_variable([3, 3, in_channels, depth], name='weight_3x3_1')
            weight_3_2 = weight_variable([3, 3, in_channels, depth], name='weight_3x3_2')
            weight_3_3 = weight_variable([3, 3, in_channels, depth], name='weight_3x3_3')

            conv_1x1 = tf.nn.conv2d(inputs, weight_1, [1, 1, 1, 1], padding='SAME', name='conv_1x1')
            conv_3x3_1 = tf.nn.atrous_conv2d(inputs, weight_3_1, rate=atrous_rates[0], padding='SAME', name='conv_3x3_1')
            conv_3x3_2 = tf.nn.atrous_conv2d(inputs, weight_3_2, rate=atrous_rates[0], padding='SAME', name='conv_3x3_2')
            conv_3x3_3 = tf.nn.atrous_conv2d(inputs, weight_3_3, rate=atrous_rates[0], padding='SAME', name='conv_3x3_3')

            conv_1x1 = tf.nn.relu(batch_norm(conv_1x1, is_training), name='relu_1x1')
            conv_3x3_1 = tf.nn.relu(batch_norm(conv_3x3_1, is_training), name='relu_3x3_1')
            conv_3x3_2 = tf.nn.relu(batch_norm(conv_3x3_2, is_training), name='relu_3x3_2')
            conv_3x3_3 = tf.nn.relu(batch_norm(conv_3x3_3, is_training), name='relu_3x3_3')

        # (b)
        with tf.name_scope("aspp_b"):
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
            weight_b = weight_variable([1, 1, in_channels, depth], name='weight_1x1_b')
            image_level_features = tf.nn.conv2d(image_level_features, weight_b, [1, 1, 1, 1], padding='SAME', name='conv_1x1_b')
            image_level_features = tf.nn.relu(batch_norm(image_level_features, is_training), name='relu_1x1_b')
            image_level_features = tf.image.resize_bilinear(image_level_features, input_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=-1, name='concat')

        weight_a_b = weight_variable([1, 1, depth*5, depth], name='weight_1_a_b')
        net = tf.nn.conv2d(net, weight_a_b, [1, 1, 1, 1], padding='SAME', name='conv_a_b')
        net = tf.nn.relu(batch_norm(net, is_training), name='relu_a_b')

    return net



def deeplab_v3_plus(inputs, is_training, output_stride, pre_trained_model):

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        logits, end_points = resnet_v2.resnet_v2_101(inputs, num_classes=None, is_training=is_training, global_pool=False, output_stride=output_stride)

    if is_training:
        exclude = ['resnet_v2_101' + '/logits', 'global_step']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        tf.train.init_from_checkpoint(pre_trained_model, {v.name.split(':')[0]: v for v in variables_to_restore})

    net = end_points['resnet_v2_101' + '/block4']
    encoder_output = aspp(net, output_stride, is_training)

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        with tf.name_scope('low_level_features'):
            low_level_features = end_points['resnet_v2_101' + '/block1/unit_3/bottleneck_v2/conv1']
            in_channels = low_level_features.get_shape().as_list()[-1]
            low_level_shape = tf.shape(low_level_features)
            weight_1x1_low_level = weight_variable([1, 1, in_channels, 48], name='weight_1x1_low_level')
            conv_1x1_low_level = tf.nn.conv2d(low_level_features, weight_1x1_low_level, [1, 1, 1, 1], padding='SAME', name='conv_1x1_low_level')
            conv_1x1_low_level = tf.nn.relu(batch_norm(conv_1x1_low_level, is_training), name='relu_1x1_low_level')

            low_level_features_size = low_level_shape[1:3]

        with tf.name_scope("upsamling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')

            net = tf.concat([net, low_level_features], axis=-1, name='concat')
            weight_3x3_upsamle_1 = weight_variable([3, 3, net.get_shape().as_list()[-1], 256], name='weight_3x3_upsamle_1')
            weight_3x3_upsamle_2 = weight_variable([3, 3, 256, 256], name='weight_3x3_upsamle_2')
            weight_3x3_upsamle_3 = weight_variable([3, 3, 256, CLASSES], name='weight_3x3_upsamle_3')

            net = tf.nn.conv2d(net, weight_3x3_upsamle_1, [1, 1, 1, 1], padding='SAME', name='conv_3x3_upsamle_1')
            net = tf.nn.relu(batch_norm(net, is_training), name='conv_3x3_relu_1')
            net = tf.nn.conv2d(net, weight_3x3_upsamle_2, [1, 1, 1, 1], padding='SAME', name='conv_3x3_upsamle_2')
            net = tf.nn.relu(batch_norm(net, is_training), name='conv_3x3_relu_2')
            net = tf.nn.conv2d(net, weight_3x3_upsamle_3, [1, 1, 1, 1], padding='SAME', name='conv_3x3_upsamle_3')

            logits = tf.image.resize_bilinear(net, tf.shape(inputs)[1:3], name='upsample_2')


    return logits


if __name__ == '__main__':

    inputs = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 3], name='x_input')


    #inputs = tf.constant(1.0, shape=[8, None, None, 3])
    b = deeplab_v3_plus(inputs, is_training=True, output_stride=16, pre_trained_model=PRETRAINED_MODEL_PATH)

    print(b)






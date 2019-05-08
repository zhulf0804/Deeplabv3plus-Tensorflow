"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

CLASSES = 21
PRETRAINED_MODEL_PATH = './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt'

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


def deeplab_v3_plus_generator(num_classes,
                              output_stride,
                              base_architecture,
                              pre_trained_model,
                              batch_norm_decay,
                              data_format='channels_last'):
  """Generator for DeepLab v3 plus models.

  Args:
    num_classes: The number of possible classes for image classification.
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    base_architecture: The architecture of base Resnet building block.
    pre_trained_model: The path to the directory that contains pre-trained models.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
      Only 'channels_last' is supported currently.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the DeepLab v3 model.
  """
  if data_format is None:
    # data_format = (
    #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    pass

  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY

  if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
    raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")

  if base_architecture == 'resnet_v2_50':
    base_model = resnet_v2.resnet_v2_50
  else:
    base_model = resnet_v2.resnet_v2_101

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # tf.logging.info('net shape: {}'.format(inputs.shape))
    # encoder
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      logits, end_points = base_model(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride)

    if is_training:
      exclude = [base_architecture + '/logits', 'global_step']
      variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
      tf.train.init_from_checkpoint(pre_trained_model,
                                    {v.name.split(':')[0]: v for v in variables_to_restore})

    inputs_size = tf.shape(inputs)[1:3]
    net = end_points[base_architecture + '/block4']
    encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

    with tf.variable_scope("decoder"):
      with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
          with tf.variable_scope("low_level_features"):
            low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
            low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                   [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

          with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

    return logits

  return model


def deeplabv3_plus_model_fn(features):
  """Model function for PASCAL VOC."""


  network = deeplab_v3_plus_generator(21,
                                      16,
                                      'resnet_v2_101',
                                      './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt',
                                      _BATCH_NORM_DECAY)

  logits = network(features, True)

  return logits

if __name__ == '__main__':

    inputs = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 3], name='x_input')


    #inputs = tf.constant(1.0, shape=[8, None, None, 3])
    b = deeplabv3_plus_model_fn(inputs)

    print(b)

#!/usr/bin/env python


import lasagne

# Add layers needed for other pretrained models here
from lasagne.layers import InputLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax

import pickle

from layers.mylayers import RGBtoBGRLayer

# Path to pretrained models
pathPretrained = '/data/lisatmp3/romerosa/pretrained/'


def build_VGG19(inputSize, input_var=None,
                pathVGG19=pathPretrained+'vgg19.pkl',
                last_layer='fc7_dropout', trainable=False):
    """
    Construct VGG19 convnet
    """

    net = {}
    net['input'] = InputLayer((None, inputSize[0], inputSize[1], inputSize[2]),
                              input_var)
    net['bgr'] = RGBtoBGRLayer(net['input'])
    net['conv1_1'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3,
                               pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3,
                               pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3,
                               pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3,
                               pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1,
                               flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1,
                               flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3,
                               pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3,
                               pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7_dropout'],
                            num_units=1000,
                            nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    pretrained_values = pickle.load(open(pathVGG19))['param values']

    nlayers = len(lasagne.layers.get_all_params(net[last_layer]))

    lasagne.layers.set_all_param_values(net[last_layer],
                                        pretrained_values[:nlayers])

    # Do not train
    if not trainable:
        freezeParameters(net)

    return net[last_layer]


def build_VGG16(inputSize, input_var=None,
                pathVGG16=pathPretrained+'vgg16.pkl',
                last_layer='fc7_dropout', trainable=False):
    """
    Construct VGG16 convnet
    """

    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var)
    net['bgr'] = RGBtoBGRLayer(net['input'])
    net['conv1_1'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3,
                               pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3,
                               pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3,
                               pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3,
                               pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3,
                               pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3,
                               pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    pretrained_values = pickle.load(open(pathVGG16))['param values']

    nlayers = len(lasagne.layers.get_all_params(net[last_layer]))

    lasagne.layers.set_all_param_values(net[last_layer],
                                        pretrained_values[:nlayers])

    # Do not train
    if not trainable:
        freezeParameters(net)

    return net[last_layer]


def freezeParameters(net):
    all_layers = net.values()
    for net_layer in all_layers:
        layer_params = net_layer.get_params()
        for l in layer_params:
            try:
                net_layer.params[l].remove('trainable')
            except KeyError:
                pass

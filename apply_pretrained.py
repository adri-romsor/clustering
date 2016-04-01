#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

import lasagne

import argparse

import build_pretrained as pretrained
import loader.data_polyps as data_polyps

# Path to pretrained models
pathPretrained = '/data/lisatmp3/romerosa/pretrained/'
debug = False


# Build pre-trained model
def buildModel(input_var, model, imSize, **kwargs):
    if model == 'vgg19':
        network = pretrained.build_VGG19(imSize, input_var, **kwargs)
    elif model == 'vgg16':
        network = pretrained.build_VGG16(imSize, input_var, **kwargs)
    else:
        print("Unrecognized model type %r." % model)
        return

    return network


# Iterate over minibatches
def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


# Main program
def execute(model, dataset, **kwargs):
    # Load the dataset
    print("Loading data...")
    if dataset == 'polyps':
        imSize, _, _, valid, _ = data_polyps.load_data(crop=True)
    else:
        print("Unknown dataset")
        return

    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')

    # Create neural network model by loading pre-trained model
    # specified in the first parameter
    print("Building model...")
    network = buildModel(input_var, model, imSize, **kwargs)

    # Create expression for predicting outputs
    print("Building and compiling prediction functions...")
    embedding = lasagne.layers.get_output(network, deterministic=True)
    emb_fn = theano.function([input_var], embedding)

    out_emb = []
    for batch in iterate_minibatches(valid[0], 50, shuffle=False):
        inputs = batch
        emb = emb_fn(inputs)
        out_emb.append(emb)

    out_emb = np.ma.concatenate(out_emb)

    print("Number of output embeddings {} with size {}".format(
                out_emb.shape[0], out_emb.shape[1]))


def main():
    parser = argparse.ArgumentParser(description='Load supervised-pre-trained'
                                     'model and extract embedded'
                                     'features for a given dataset.')
    parser.add_argument('model',
                        default='vgg19',
                        help='Pre-trained model.')
    parser.add_argument('dataset',
                        help='Dataset.')
    parser.add_argument('--layer',
                        '-l',
                        type=str,
                        default=None,
                        help='Optional. str to indicate the layer from'
                        'which we want to extract features.')

    args = parser.parse_args()

    kwargs = {}
    if args.layer is not None:
        kwargs['last_layer'] = args.layer
    execute(args.model, args.dataset, **kwargs)


if __name__ == '__main__':
    main()

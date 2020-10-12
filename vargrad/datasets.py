# Taken from JAX examples

"""Datasets used in examples."""

import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np
from scipy.io import loadmat


_DATA = "~/.jax_example_data/"


def _download(url, filename):
    """
    Download a url to a file in the JAX data temp directory.
    """
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        print('downloading...')
        urllib.request.urlretrieve(url, out_file)
        print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
    """
    Flatten all but the first dimension of an ndarray.
    """
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """
    Create a one-hot encoding of x of size k.
    """
    return np.array(x[:, None] == np.arange(k), dtype)


def omniglot_raw():
    """
    Download and parse the raw OMNIGLOT dataset.
    """
    base_url = "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat"

    def fetch_item(filename, item):
        data_dict = loadmat(filename)
        return data_dict[item].T

    filename = 'omniglot.mat'
    _download(base_url, filename)

    train_images = fetch_item(path.join(_DATA, filename), 'data')
    test_images = fetch_item(path.join(_DATA, filename), 'testdata')

    return train_images, test_images


def omniglot(permute_train=False):
    """
    Download, parse and process OMNIGLOT data to unit scale and one-hot labels.
    """
    train_images, test_images = omniglot_raw()

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]

    return np.round(train_images), np.round(test_images)


def mnist_raw():
    """
    Download and parse the raw binarised MNIST dataset.
    """
    base_url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'

    data = []
    for filename in ['binarized_mnist_train.amat', 'binarized_mnist_test.amat']:
        _download(base_url + filename, filename)
        data.append(np.loadtxt(_DATA + filename))

    return tuple(data)


def mnist(permute_train=False):
    """
    Download, parse and process binarized MNIST data to unit scale and one-hot labels.
    """
    train_images, test_images = mnist_raw()

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]

    return train_images, test_images


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-member

from __future__ import absolute_import
from __future__ import print_function

import logging
import logging.config
import os
import sys
from configparser import ConfigParser

import numpy as np
import tensorflow as tf

from utils import print_runtime_statistics
from utils import UnsupportedModelError


class RidgeRegressor():
    """The model which performs ridge regression."""

    def __init__(self, d, alpha):
        """Initializes model parameters.

        Args:
            d: int, the feature dimension.
            alpha: float, the regularization coefficient.
        """
        random_uniform = tf.initializers.RandomUniform(-0.05, 0.05)
        self.w = tf.Variable(random_uniform([d]), name='weight')
        self.alpha = alpha

    @tf.function
    def predict(self, x):
        """Outputs the tensor for model prediction.

        Args:
            x: a 2D Tensor with shape (ANY, d), where d is the feature
                dimension.
        Returns:
            A 1D Tensor with length ANY.
        """
        w = self.w
        return tf.reduce_sum(x * w, axis=1)

    @tf.function
    def loss(self, y_predicted, y_true):
        """Outputs the tensor for loss.

        Args:
            y_predicted: a 1D Tensor with any length.
            y_true: a 1D Tensor with any length, but should have the same
                length as `y_predicted`.
        """
        w = self.w
        l2 = tf.reduce_sum(self.alpha * (w ** 2))
        return tf.reduce_mean((y_predicted - y_true) ** 2) + l2


def optimize_in_tf(feature_matrix, label_array, lr_scheduler, args, stat,
                   model_type):
    """Runs optimization in tensorflow.

    Args:
        feature_matrix: a 2D numpy array of floats with shape N x d.
        label_array: an 1D numpy array of floats with shape N.
        lr_scheduler: a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
            function arguments:
                init_lr: float, the intial learning rate;
                t: int, current iteration number, starts from 0, i.e. we have
                    t=0 for first SGD update.
        args: a 'class-like' object, the content parsed from commandline. It
            elements can be directly accessed in a way like 'args.num_epoch'.
        stat: a dict, which maps names to relevant precomputed ridge regression
            statistics.
        model_type: a str specifies the model type. Supported types are:
            'ridge_regression',
    Raises:
        UnsupportedModelError if 'model_type' not supported.
    """
    x_train = tf.Variable(feature_matrix, dtype='float32')
    y_train = tf.Variable(label_array, dtype='float32')
    n = stat['num_sample']
    d = stat['feature_dimension']

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(
        buffer_size=n,
        reshuffle_each_iteration=True)
    dataset = dataset.repeat(args.num_epoch)

    # Runs optimization in tensorflow
    if model_type == 'ridge_regression':
        model = RidgeRegressor(d, args.alpha)
    else:
        raise UnsupportedModelError('unsupported model type "%s"' % model_type)

    optimizer = tf.optimizers.SGD(args.init_lr)

    min_loss = 1e6
    for t, (x, y) in enumerate(dataset):
        learning_rate = lr_scheduler(args.init_lr, t)
        optimizer.lr.assign(learning_rate)

        with tf.GradientTape() as grad_tape:
            x_ = tf.reshape(x, shape=(1, d))
            y_predicted = model.predict(x_)
            loss = model.loss(y_predicted, y)

        gradients = grad_tape.gradient(loss, model.w)
        optimizer.apply_gradients([(gradients, model.w)])

        # Gets loss over the whole dataset
        if t % 100 == 0 or t % n == n - 1:
            y_predicted = model.predict(x_train)
            loss = model.loss(y_predicted, y_train)
            min_loss = min(min_loss, loss)
            if t % n == n - 1:
                logging.info('========== epoch %d ==========', t // n)
                logging.info('min_loss: %.6f', min_loss)

            logging.info('loss: %.6f', loss)

        # Prints distance of each dimension in rotated space, where
        # eigen_values satisfy:
        #   lambda_0 >= lambda_1 >= ... >= lambda_{d-1}
        if t % 100 == 0:
            print_runtime_statistics(t, stat, model.w.numpy(), loss)

    logging.info('========== summary ==========')
    logging.info('w = %s', str(model.w.numpy().tolist()))
    logging.info('min loss = %.6lf', min_loss)
    logging.info('stat.optimum = %s', stat['optimum'].tolist())
    logging.info('stat.optimal_loss = %s', stat['optimal_loss'])

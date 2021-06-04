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

from tf_model_trainer import optimize_in_tf
from utils import load_input
from utils import parse_argument
from utils import parse_lr_schedule_conf
from utils import print_runtime_statistics
from utils import UnsupportedFrameworkError


def get_stat(feature_matrix, label_array, alpha):
    """Gets necessary statistics of ridge regression.

    Necessary statistics of ridge regression:
        * The rotation matrix P which transforms the hessian matrix into a
          diagonal matrix, with descending eigenvalues, i.e. H = P^{-1} D P,
          D = {lambda_i}_{i=0}^{d-1}, where lambda_0 >= lambda_2 >= ... >=
          lambda_{d-1}
        * Optimum ~w* in the rotated space
        * Eigenvalues of hessian matrix H, in descending orders

    Args:
        feature_matrix: a 2D numpy array of floats with shape N x d.
        label_array: an 1D numpy array of floats with shape N.
        alpha: float, the regularization coefficient.

    Returns:
        A dict which maps statistics name to its value, including
            "rotation_matrix": a 2D numpy array of floats with shape d x d;
            "rotated_optimum": a 1D numpy array of floats with shape d;
            "eigen_values": a 1D numpy array of floats with shape d, the
                    eigen_values for the hessian_matrix in descending order;
            "optimal_loss": optimum loss of ridge regression
    """
    n, d = feature_matrix.shape
    x = feature_matrix

    # H = 2 * X^T X / n + 2 * alpha I
    hessian_matrix = 2 * np.matmul(np.transpose(x), x) / n + 2 * alpha * np.identity(d)

    # H = P^T D P, eigenvalues are in ascending order by default
    eigen_values, eigen_vectors = np.linalg.eigh(hessian_matrix)
    rotation_matrix = np.transpose(eigen_vectors)

    # Reorders eigenvalues, in descending order
    indices = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[indices]
    rotation_matrix = rotation_matrix[indices, :]

    logging.debug('eigen values: %s', str(eigen_values.tolist()))

    p = rotation_matrix
    # In the rotated space:
    #   ~w*_i = P_i X^T y / eigen_i
    #         (= D^{-1} (P X^T y))
    # In the original space:
    #   w* = P^T ~w*
    #     (= P^T D^{-1} P X^T y = H^{-1} X^T y)
    y = label_array
    rotated_optimum = np.matmul(p, 2 * np.matmul(np.transpose(x), y) / n)
    rotated_optimum = np.divide(rotated_optimum, eigen_values)
    optimum = np.matmul(np.transpose(p), rotated_optimum)

    logging.debug('rotated optimum: %s', str(rotated_optimum.tolist()))
    logging.debug('optimum: %s', str(optimum.tolist()))

    loss_without_r = np.sum((y - np.matmul(x, optimum)) ** 2) / n
    loss = loss_without_r + alpha * np.sum(optimum ** 2)
    # loss_without_r /= n
    # loss /= n
    logging.debug('loss without regularization term: %.6lf', loss_without_r)
    logging.debug('loss: %.6lf', loss)

    # Returns statistics
    stat = {
        'rotation_matrix': rotation_matrix,
        'rotated_optimum': rotated_optimum,
        'optimum': optimum,
        'eigen_values': eigen_values,
        'optimal_loss': loss}
    return stat


def optimize_in_np(feature_matrix, label_array, lr_scheduler, args, stat):
    """Runs optimization in numpy & manually computed gradients.

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
    """
    n = stat['num_sample']
    d = stat['feature_dimension']
    x_train = feature_matrix
    y_train = label_array

    def shuffle(x_train, y_train):
        indices = np.arange(n)
        np.random.shuffle(indices)
        return x_train[indices, :], y_train[indices]

    x_train, y_train = shuffle(x_train, y_train)

    # Runs ridge regression
    min_loss = 1e6
    w = np.random.uniform(-0.05, 0.05, d)

    for t in range(args.num_epoch * n):
        if t % n == 0:
            x_train, y_train = shuffle(x_train, y_train)

        # Gets sample
        sample_id = t % n
        x = x_train[sample_id]
        y = y_train[sample_id]

        # Gets prediction
        y_predicted = np.sum(w * x)

        # Gets gradients
        alpha = args.alpha
        grad = 2 * (y_predicted - y) * x + 2 * alpha * w

        # Applies gradients
        learning_rate = lr_scheduler(args.init_lr, t)
        w -= learning_rate * grad

        # Gets loss over the whole dataset
        if t % 100 == 0 or t % n == n - 1:
            y_predicted = np.dot(x_train, w)
            loss = np.mean((y_predicted - y_train) ** 2)
            loss += alpha * np.sum(w ** 2)
            min_loss = min(min_loss, loss)
            if t % n == n - 1:
                logging.info('========== epoch %d ==========', t // n)
                logging.info('min_loss: %.6f', min_loss)

            logging.info('learning rate: %.10f', learning_rate)
            logging.info('loss: %.10f', loss)

        # Prints distance of each dimension in rotated space, where
        # eigen_values satisfy:
        #   lambda_0 >= lambda_1 >= ... >= lambda_{d-1}
        if t % 100 == 0:
            print_runtime_statistics(t, stat, w, loss)

    logging.info('========== summary ==========')
    logging.info('w = %s', str(w.tolist()))
    logging.info('min loss = %.12lf', min_loss)
    logging.info('stat.optimum = %s', stat['optimum'].tolist())
    logging.info('stat.optimal_loss = %s', stat['optimal_loss'])


def main():
    """Uses ridge regression to analyze SGD with learning rate schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Parses conf file for learning rate schedule
    logging.info('lr_schedule_conf_file = %s', args.lr_schedule_conf_file)
    lr_schedule_conf = ConfigParser()
    lr_schedule_conf.read(args.lr_schedule_conf_file)
    lr_scheduler = parse_lr_schedule_conf(lr_schedule_conf)

    # Controls pseudorandom behavior
    if args.pseudo_random:
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        np.random.seed(1)
        tf.random.set_seed(1)

    # Loads input data and converts to numpy
    feature_matrix, label_array = load_input(args.train_data)
    n, d = feature_matrix.shape
    logging.info('load input data complete, %d samples, %d dims', n, d)

    # Calculates statistics for analysis, e.g. optimal point
    stat = get_stat(feature_matrix, label_array, args.alpha)
    stat['num_sample'] = n
    stat['feature_dimension'] = d
    logging.info('compute statistics complete')

    # Runs optimization
    if args.framework == 'tensorflow':
        optimize_in_tf(feature_matrix,
                       label_array,
                       lr_scheduler,
                       args,
                       stat,
                       model_type='ridge_regression')
    elif args.framework == 'numpy':
        optimize_in_np(feature_matrix, label_array, lr_scheduler, args, stat)
    else:
        raise UnsupportedFrameworkError(
            'framework "%s" not supported' % args.framework)


if __name__ == '__main__':
    main()

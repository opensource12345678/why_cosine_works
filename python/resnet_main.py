#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=self-assigning-variable
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=literal-comparison

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import logging.config
import os
import sys
import textwrap
from configparser import ConfigParser

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras            # pylint: disable=import-error

from resnet import ResNet18
from resnet import ResNet50
from resnet import ResNet101
from resnet import ResNet152

from utils import parse_lr_schedule_conf
from utils import UnsupportedDatasetError
from utils import UnsupportedModelError


def parse_argument(sys_argv):
    """Parses arguments from command line.

    Args:
        sys_argv: the list of arguments (strings) from command line.

    Returns:
        A struct whose member corresponds to the required (optional) variable.
        For example,
        ```
        args = parse_argument(['main.py' '--input', 'a.txt', '--num', '10'])
        args.input       # 'a.txt'
        args.num         # 10
        ```
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Deep learning models for image classification')

    # Training parameters
    parser.add_argument(
        '--dataset', type=str, required=True,
        help=textwrap.dedent(
            '''
            Supported options:
              1) "cifar10"
              2) "cifar100"
            '''))

    parser.add_argument(
        '--num_epoch', type=int, required=True,
        help='The number of epochs')
    parser.add_argument(
        '--batch_size', type=int, required=True, default=1,
        help='Batch size')
    parser.add_argument(
        '--init_lr', type=float, required=True,
        help='Initial learning rate')
    parser.add_argument(
        '--l2_coefficient', type=float, default=0.0,
        help='Coefficient for L2 term')
    parser.add_argument(
        '--lr_schedule_conf_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The config file name for different learning rate schedules.
            Its general format is as follows,
            ```
            [general]
            type = {type}

            [hyperparams]
            ...
            ```

            Supported types of learning rate schedules are as follows,
                * "inverse_time_decay":
                    During iteration t, learning rate is
                        `init_lr / (1 + init_lr * t * lambda)`
                    here lambda is a hyperparameter for this schedule.
                    Iteration t starts from 0.
                    Config example:
                        ```
                        [general]
                        type = inverse_time_decay

                        [hyperparams]
                        lambda = 0.1
                        ```

                * "piecewise_constant":
                    Specifies the starting point of each interval s_i (i>0),
                    learning rate will be,
                        `init_lr * c_i`
                    if t in [s_i, s_{i+1}). Here c_i is the factor of this
                    interval. s_0=0 and s_{n+1}=+oo by default.
                    Config example:
                        ```
                        [general]
                        type = piecewise_constant

                        [hyperparams]
                        starting_points = 100, 500, 1000
                        factors = 0.1, 0.01, 0.001
                        ```

                * "cosine_decay":
                    [2016] SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
                    (https://arxiv.org/pdf/1608.03983.pdf)

                    Learning rate decays in each segment [0, t_max]. t_max =
                    t_0, t_cur=0 initially. After the end of each segment,
                    expand the length of that segment by multiplying `t_mul`,
                    i.e. t_max *= t_mul, then increase `t_cur` by 1, i.e. t_cur
                    += 1.
                        ```python
                        cos_decay = 0.5 * (1 + cos(pi * t_cur / t_max))
                        lr = min_lr + (init_lr - min_lr) * cos_decay
                        ```

                    Config example:
                        ```
                        [general]
                        type = cosine_decay

                        [hyperparams]
                        t_0 = 100
                        t_mul = 1.0
                        min_lr = 0.00001

                * "exponential_decay":
                    During interation t, which starts from 0 initially,
                        `lr = init_lr * (decay_rate ** (t / decay_step))`
                    Note that the division here `t / decay_step` is integer
                    division.

                    Config example:
                        ```
                        [general]
                        type = exponential_decay

                        [hyperparams]
                        decay_step = 1
                        decay_rate = 0.9999
                        ```

                * "piecewise_inverse_time":
                    Specifies the starting point of each interval s_i (i>=0),
                    learning rate will be,
                        `init_lr / (a_i * (t - s_i) + b_i)`
                    if t in [s_i, s_{i+1}). Here a_i and b_i is the
                    hyperparameter for interval. It is required that s_1=0.
                    s_{n+1}=+oo.

                    Config example:
                        ```
                        [general]
                        type = piecewise_constant

                        [hyperparams]
                        starting_points = 0, 100, 500, 1000
                        a = 1.0, 2.0, 4.0, 8.0
                        b = 0.0, 0.0, 0.0, 0.0
                        ```
            '''))

    # Model parameters
    parser.add_argument(
        '--model', type=str, default='resnet18',
        help=textwrap.dedent(
          '''
          Supported models:
            * resnet18
            * resnet50
            * resnet101
            * resnet152
          '''))

    # Debug parameters
    parser.add_argument(
        '--pseudo_random', const=True, default=False, nargs='?',
        help='A global option to make all random operations deterministic')
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')
    parser.add_argument(
        '--num_sample_for_debug', type=int, default=-1,
        help=('The number of train/test samples we will pick for debugging'
              ', where other sample will be ignored. If set to -1'
              ' (default), then all samples will be used'))

    # Test parameters
    parser.add_argument(
        '--val', const=True, default=False, nargs='?',
        help='Turn on validation mode')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def optimize_in_tf(x_train_np, y_train_np, x_test_np, y_test_np, num_class,
                   l2_coeff, lr_scheduler, args, stat, model_type):
    """Runs optimization in tensorflow.

    Args:
        x_train_np: a 4D numpy array of floats, which represent the training
            set of images (sample_id, row_id, column id, channel id).
        y_train_np: a 2D numpy array of ints, which represents the training set
            of labels (sample_id, class_id).
        x_test_np: a 4D numpy array of floats, which represent the test set of
            images (sample_id, row_id, column id, channel id).
        y_test_np: a 2D numpy array of ints, which represents the test set of
            labels (sample_id, class_id).
        num_class: int, the number of classes.
        l2_coeff: float, coefficient for L2 regularization penalty.

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
        stat: a dict, which maps names to relevant statistics of this dataset,
            e.g. number of samples, number of classes.
        model_type: a str specifies the model type. Supported types are:
            'resnet18'

    Raises:
        UnsupportedModelError if 'model_type' not supported.
    """
    _, height, width, num_channel = x_train_np.shape

    x_train = tf.Variable(x_train_np, dtype='float32')
    y_train = tf.reshape(y_train_np, (-1,))
    x_test = tf.Variable(x_test_np, dtype='float32')
    y_test = tf.reshape(y_test_np, (-1,))

    n = stat['num_sample']
    batch_size = args.batch_size

    # Image augmentation
    def augmentation(x, y):
        x = tf.image.resize_with_crop_or_pad(x, height + 8, width + 8)
        x = tf.image.random_crop(x, [height, width, num_channel])
        x = tf.image.random_flip_left_right(x)
        return x, y

    # Prepares dataset pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(augmentation)
    train_dataset = train_dataset.shuffle(
        buffer_size=n,
        reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # Chooses model
    if model_type == 'resnet18':
        model = ResNet18(num_classes=num_class)
    elif model_type == 'resnet50':
        model = ResNet50(num_classes=num_class)
    elif model_type == 'resnet101':
        model = ResNet101(num_classes=num_class)
    elif model_type == 'resnet152':
        model = ResNet152(num_classes=num_class)
    else:
        raise UnsupportedModelError('unsupported model type "%s"' % model_type)

    # Prepares for training
    optimizer = keras.optimizers.SGD(args.init_lr)
    mean_crossent_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    crossent_loss_tracker = keras.metrics.Mean(name='crossent_loss')
    train_loss_tracker = keras.metrics.Mean(name='train_loss')
    train_accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_loss_tracker = keras.metrics.Mean(name='test_loss')
    test_accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    # Training
    @tf.function
    def l2_loss():
        weights = model.trainable_variables
        return tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_coeff

    @tf.function
    def mean_crossent_loss_with_l2(y_batch, predicted_y_prob):
        loss = mean_crossent_loss(y_batch, predicted_y_prob)
        return loss + l2_loss()

    if l2_coeff < 1e-8:
        get_mean_loss = mean_crossent_loss
    else:
        get_mean_loss = mean_crossent_loss_with_l2

    @tf.function
    def train_on_batch(x_batch, y_batch):
        with tf.GradientTape() as grad_tape:
            predicted_y_prob = model(x_batch, training=True)
            loss = get_mean_loss(y_batch, predicted_y_prob)
        gradients = grad_tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        crossent_loss_tracker(loss - l2_loss())
        train_loss_tracker(loss)
        train_accuracy_tracker(y_batch, predicted_y_prob)
        return loss

    @tf.function
    def test_on_batch(x_batch, y_batch):
        predicted_y_prob = model(x_batch)
        loss = get_mean_loss(y_batch, predicted_y_prob)
        test_loss_tracker(loss)
        test_accuracy_tracker(y_batch, predicted_y_prob)

    t = 0
    for epoch_id in range(args.num_epoch):
        logging.info('========== epoch %d ==========', epoch_id)
        for x_batch, y_batch in train_dataset:
            learning_rate = lr_scheduler(args.init_lr, t)
            t += 1
            optimizer.lr.assign(learning_rate)

            batch_loss = train_on_batch(x_batch, y_batch)

            if t % (50 * args.num_epoch // batch_size) == 0:
                logging.info('learning rate: %.10lf', learning_rate)
                logging.info('batch loss: %.10lf', batch_loss.numpy())

        # Tests at the end of every epoch
        for x_batch, y_batch in test_dataset:
            test_on_batch(x_batch, y_batch)

        # Prints summarized statistics at the end of every epoch
        logging.info(
            'End of epoch %d'
            ', crossentropy loss only: %.10lf'
            ', moving-average train loss: %.10lf'
            ', moving-average train accuracy: %.6lf'
            ', test loss: %.10lf, test accuracy: %.6lf',
            epoch_id,
            crossent_loss_tracker.result(),
            train_loss_tracker.result(),
            train_accuracy_tracker.result(),
            test_loss_tracker.result(),
            test_accuracy_tracker.result())

        crossent_loss_tracker.reset_states()
        train_loss_tracker.reset_states()
        train_accuracy_tracker.reset_states()
        test_loss_tracker.reset_states()
        test_accuracy_tracker.reset_states()

    # Gets training loss for final result. Notice it is not the moving loss
    # anymore, but the real loss over the whole dataset.
    logging.info('========== summary ==========')
    for x_batch, y_batch in train_dataset:
        test_on_batch(x_batch, y_batch)

    logging.info('Summary: train loss: %.10lf, train accuracy: %.6lf',
                 test_loss_tracker.result(),
                 test_accuracy_tracker.result())


def main():
    """Uses deep learning models to analyze SGD with learning rate schedule."""
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
    if args.dataset == 'cifar10':
        cifar10 = keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_sample = x_train.shape[0]
        num_class = 10

        mean = (0.4914, 0.4822, 0.4465)     # Mean per channel in training set
        stddev = (0.2023, 0.1994, 0.2010)
        x_train = (x_train / 255.0 - mean) / stddev
        x_test = (x_test / 255.0 - mean) / stddev
    elif args.dataset == 'cifar100':
        cifar100 = keras.datasets.cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_sample = x_train.shape[0]
        num_class = 100

        mean = (0.5071, 0.4867, 0.4408)
        stddev = (0.2675, 0.2565, 0.2761)
        x_train = (x_train / 255.0 - mean) / stddev
        x_test = (x_test / 255.0 - mean) / stddev
    else:
        raise UnsupportedDatasetError('unsupported dataset "%s"' % args.dataset)

    stat = {}
    stat['num_sample'] = num_sample
    stat['num_class'] = num_class

    # Just for debugging in machines with small memory
    if args.num_sample_for_debug > 0:
        n = args.num_sample_for_debug
        stat['num_sample'] = n
        x_train, y_train = x_train[:n], y_train[:n]
        x_test, y_test = x_test[:n], y_test[:n]

    # Validation mode
    if args.val:
        n = int(x_train.shape[0] * 0.9 + 0.5)
        stat['num_sample'] = n
        x_test, y_test = x_train[n:], y_train[n:]
        x_train, y_train = x_train[:n], y_train[:n]

    logging.info('x_train.shape = %s', str(x_train.shape))
    logging.info('y_train.shape = %s', str(y_train.shape))
    logging.info('load input data complete, %d samples', num_sample)

    # Runs optimization
    optimize_in_tf(
        x_train,
        y_train,
        x_test,
        y_test,
        num_class,
        args.l2_coefficient,
        lr_scheduler,
        args,
        stat,
        model_type=args.model)


if __name__ == '__main__':
    main()

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
import math


class Error(Exception):
    """Root for all errors."""

class ParseError(Error):
    """Unable to parse input data."""

def get_activation_scheduler(conf, return_by_t=True):
    """Parses the config for activation schedulers.

    Before the specified activation point, we use constant scheduling by
    default. Right after the activation point, the chosen scheduler is
    activated, and iteration number is counted since that point.

    Activation scheduling is always an additional component of other
    schedulers.  So its interface is different from other common schedulers.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
            Following configure information is used. For example,
                ```
                [hyperparams]
                activation_point = 100
                ```
    Returns:
        if return_by_t=True, returns a function which adjusts t according to
        activation settings,
            ```
            new_t = lr_scheduler(t)
            ```
        function arguments:
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.

        For example, with activation point being 8, the returned new_t will
            be
            ```
            t       0  1  2  3  4  5  6  7  8  9  10  11  12  13
            new_t   0  0  0  0  0  0  0  0  0  1  2   3   4   5
                                            ^
                                            activated
                    |--- use constant lr ---|
            ```

        Otherwise, returns a function which specifies whether the scheduler is
        activated,
            ```
            activated = lr_scheduler(t)
            # activated: True or False
            ```
    Raises:
        ParseError if hyperparameters are invalid.
    """
    if not conf.has_option('hyperparams', 'activation_point'):
        activation_point = 0
        logging.info('lr schedule: activation option = False')
    else:
        activation_point = conf.getint('hyperparams', 'activation_point')
        logging.info('lr schedule: activation option = True')
        logging.info('lr schedule: activation point = %d', activation_point)

    if return_by_t:
        activation_scheduler = lambda t: max(0, t - activation_point)
    else:
        activation_scheduler = lambda t: t >= activation_point
    return activation_scheduler


def get_restart_scheduler(conf):
    """Parses the config for restart schedulers.

    Restart the chosen scheduler when at the specified restarting points.

    Restart scheduling is always an additional component of other schedulers.
    So its interface is different from other common schedulers.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
            Following configure information is used. For example,
                ```
                [hyperparams]
                restarting_points = 0, 10000        # Assumes to be sorted
                ```
    Returns:
        a function which adjusts t according to restart settings,
            ```
            new_t = lr_scheduler(t)
            ```
        function arguments:
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.

        For example, with restarting point being [3, 5, 10], the returned new_t
            will be
            ```
            t       0  1  2  3  4  5  6  7  8  9  10  11  12  13
            new_t   0  1  2  0  1  0  1  2  3  4  0   1   2   3
                             ^     ^              ^
                        restart  restart        restart
            ```
    Raises:
        ParseError if hyperparameters are invalid.
    """
    if not conf.has_option('hyperparams', 'restarting_points'):
        restart_point_list = []
        logging.info('lr schedule: restart = False')
    else:
        raw_point_list = conf.get('hyperparams', 'restarting_points')
        restart_point_list = [int(p) for p in raw_point_list.split(',')]
        logging.info('lr schedule: restart = True')
        logging.info('lr schedule: restarting_points = %s',
                     str(restart_point_list))

    def restart_scheduler(t):
        new_t = t
        for restart_point in restart_point_list:
            if t >= restart_point:
                new_t = t - restart_point
        return new_t

    return restart_scheduler


def get_inverse_time_decay_scheduler(conf):
    """Parses the config for learning rate schedule.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    lambda_ = conf.getfloat('hyperparams', 'lambda')
    logging.info('lr schedule: type = inverse_time_decay_scheduler')
    logging.info('lr schedule: lambda = %.6lf', lambda_)

    activation_scheduler = get_activation_scheduler(conf)
    restart_scheduler = get_restart_scheduler(conf)

    def inverse_time_decay_scheduler(init_lr, t):
        t = activation_scheduler(t)
        t = restart_scheduler(t)
        return init_lr / (1 + init_lr * t * lambda_)

    return inverse_time_decay_scheduler

def get_piecewise_constant_scheduler(conf):
    """Parses the config for learning rate schedule.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    raw_start_point_list = conf.get('hyperparams', 'starting_points')
    raw_factor_list = conf.get('hyperparams', 'factors')
    start_point_list = [int(s) for s in raw_start_point_list.split(',')]
    factor_list = [float(c) for c in raw_factor_list.split(',')]
    logging.info('lr schedule: type = piecewise_constant_scheduler')
    logging.info('lr schedule: start_points = %s', str(start_point_list))
    logging.info('lr schedule: factors = %s', str(factor_list))

    # Handles errors
    if len(start_point_list) != len(factor_list):
        raise ParseError(
            'lr schedule: number of starting points differs from factors')
    if start_point_list != sorted(start_point_list):
        raise ParseError(
            'lr schedule: starting points should be sorted')

    # The generated lr scheduler
    activation_scheduler = get_activation_scheduler(conf)
    restart_scheduler = get_restart_scheduler(conf)

    def piecewise_constant_scheduler(init_lr, t):
        t = activation_scheduler(t)
        t = restart_scheduler(t)
        if not start_point_list or t < start_point_list[0]:
            return init_lr
        if t >= start_point_list[-1]:
            return init_lr * factor_list[-1]

        for i, _ in enumerate(start_point_list):
            if start_point_list[i] <= t < start_point_list[i+1]:
                return init_lr * factor_list[i]
        return init_lr

    return piecewise_constant_scheduler


def get_cosine_decay_scheduler(conf):
    """Parses the config for learning rate schedule.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    t_0 = conf.getint('hyperparams', 't_0')
    t_mul = conf.getfloat('hyperparams', 't_mul')
    min_lr = conf.getfloat('hyperparams', 'min_lr')
    power = 1
    if conf.has_option('hyperparams', 'power'):
        power = conf.getfloat('hyperparams', 'power')

    logging.info('lr schedule: type = cosine_decay_scheduler')
    logging.info('lr schedule: t_0 = %d', t_0)
    logging.info('lr schedule: t_mul = %.8lf', t_mul)
    logging.info('lr schedule: min_lr = %.8lf', min_lr)
    logging.info('lr schedule: power = %.8lf', power)

    # Handles errors
    if t_0 < 0:
        raise ParseError('lr schedule: t_0 should >= 0 (t_0 = %d)' % t_0)
    if t_mul <= 1-1e-8:
        raise ParseError('lr schedule: t_mul should >= 1 (t_mul = %.6lf)'
                         % t_mul)

    # Generated learning rate scheduler
    activation_scheduler = get_activation_scheduler(conf)
    restart_scheduler = get_restart_scheduler(conf)

    def cosine_decay_scheduler(init_lr, t):
        t = activation_scheduler(t)
        t = restart_scheduler(t)
        if min_lr > init_lr:
            raise ParseError(
                'lr schedule: min_lr should <= init_lr, (%.6lf > %.6lf)' % (
                    min_lr, init_lr))

        t_max = cosine_decay_scheduler.t_max
        t_cur = cosine_decay_scheduler.t_cur
        cos_decay_rate = 0.5 * (1 + math.cos(math.pi * t_cur / float(t_max)))
        learning_rate = min_lr + (init_lr - min_lr) * cos_decay_rate
        learning_rate = learning_rate ** power
        cosine_decay_scheduler.t_cur += 1

        # End of one segment
        if t_cur == t_max:
            cosine_decay_scheduler.t_max *= t_mul
            cosine_decay_scheduler.t_cur = 0

        return learning_rate

    cosine_decay_scheduler.t_max = t_0
    cosine_decay_scheduler.t_cur = 0

    return cosine_decay_scheduler


def get_exponential_decay_scheduler(conf):
    """Parses the config for learning rate schedule.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    decay_step = conf.getint('hyperparams', 'decay_step')
    decay_rate = conf.getfloat('hyperparams', 'decay_rate')
    logging.info('lr schedule: type = exponential_decay_scheduler')
    logging.info('lr schedule: decay_step = %d', decay_step)
    logging.info('lr schedule: decay_rate = %.6lf', decay_rate)

    # Handles errors
    if decay_step < 0:
        raise ParseError('lr schedule: decay_step should >= 0 (decay_step = %d)'
                         % decay_step)
    if decay_rate > 1+1e-8 or decay_rate < 0:
        raise ParseError('lr schedule: decay_rate should in [0, 1]'
                         ' (decay_rate = %.6lf)' % decay_rate)

    # Generated learning rate scheduler
    activation_scheduler = get_activation_scheduler(conf)
    restart_scheduler = get_restart_scheduler(conf)

    def exponential_decay_scheduler(init_lr, t):
        t = activation_scheduler(t)
        t = restart_scheduler(t)
        return init_lr * (decay_rate ** (t / decay_step))

    return exponential_decay_scheduler


def get_piecewise_inverse_time_scheduler(conf):
    """Parses the config for learning rate schedule.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate schedule.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    raw_start_point_list = conf.get('hyperparams', 'starting_points')
    raw_a_list = conf.get('hyperparams', 'a')
    raw_b_list = conf.get('hyperparams', 'b')

    start_point_list = [int(s) for s in raw_start_point_list.split(',')]
    a_list = [float(a) for a in raw_a_list.split(',')]
    b_list = [float(b) for b in raw_b_list.split(',')]

    logging.info('lr schedule: type = piecewise_inverse_time_scheduler')
    logging.info('lr schedule: start_points = %s', str(start_point_list))
    logging.info('lr schedule: a = %s', str(a_list))
    logging.info('lr schedule: b = %s', str(b_list))

    # Handles errors
    if len(start_point_list) != len(a_list):
        raise ParseError(
            'lr schedule: number of starting points differs from a')
    if len(start_point_list) != len(b_list):
        raise ParseError(
            'lr schedule: number of starting points differs from b')
    if len(start_point_list) == 0:
        raise ParseError(
            'lr schedule: starting points should have at least 1 point')
    if start_point_list[0] != 0:
        raise ParseError('lr_schedule: first starting point must be 0')
    if start_point_list != sorted(start_point_list):
        raise ParseError('lr schedule: starting points should be sorted')

    # The generated lr scheduler
    activation_indicator = get_activation_scheduler(conf, return_by_t=False)
    activation_scheduler = get_activation_scheduler(conf, return_by_t=True)
    restart_scheduler = get_restart_scheduler(conf)

    def piecewise_inverse_time_scheduler(init_lr, t):
        activated = activation_indicator(t)
        if not activated:
            return init_lr

        t = activation_scheduler(t)
        t = restart_scheduler(t)

        if t >= start_point_list[-1]:
            a = a_list[-1]
            b = b_list[-1]
            start_point = start_point_list[-1]
            return init_lr / (a * (t - start_point) + b)

        for i, _ in enumerate(start_point_list):
            if start_point_list[i] <= t < start_point_list[i+1]:
                a = a_list[i]
                b = b_list[i]
                start_point = start_point_list[i]
                return init_lr / (a * (t - start_point) + b)
        return init_lr

    if not conf.has_option('hyperparams', 'min_lr'):
        return piecewise_inverse_time_scheduler

    # Otherwise, 'min_lr' is specified.
    #   Does Linear scaling for all learning rates to make the last learning
    #   rate equal to 'min_lr'
    if not conf.has_option('hyperparams', 'num_iter'):
        raise ParseError('lr schedule: missing option "num_iter"')

    min_lr = conf.getfloat('hyperparams', 'min_lr')
    num_iter = conf.getint('hyperparams', 'num_iter')
    logging.info('lr schedule: min_lr = %.10lf', min_lr)
    logging.info('lr schedule: num_iter = %d', num_iter)

    if min_lr < 0:
        raise ParseError('lr schedule: negative min_lr %.10lf' % min_lr)
    if num_iter <= 0:
        raise ParseError('lr schedule: non-positive num_iter %d' % num_iter)

    # Return scaled learning rate scheduling
    def scaled_piecewise_inverse_time_scheduler(init_lr, t):
        lr = piecewise_inverse_time_scheduler(init_lr, t)
        # Last iteration is num_iter - 1
        old_min_lr = piecewise_inverse_time_scheduler(init_lr, num_iter - 1)

        if init_lr < min_lr:
            raise RuntimeError('init_lr < min_lr (%.10lf < %.10lf)' % (
                init_lr, min_lr))
        if init_lr < old_min_lr:
            raise RuntimeError('init_lr < old_min_lr (%.10lf < %.10lf)' % (
                init_lr, old_min_lr))

        return ((lr - old_min_lr) / (init_lr - old_min_lr) * (init_lr - min_lr)
                + min_lr)

    return scaled_piecewise_inverse_time_scheduler

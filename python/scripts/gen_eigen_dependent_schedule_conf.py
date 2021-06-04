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

import argparse
import logging
import logging.config
import math
import sys
import textwrap

import numpy as np

from utils import InvalidArgumentValueError
from utils import ParseError


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
        description='Generate proposed eigenvalue-dependent schedule conf')

    # Arguments related to input/output
    parser.add_argument(
        '--input_eigenval_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The input eigenvalue file with the following format,
                ```
                {eigenvalue_1} {density_1}
                {eigenvalue_2} {density_2}
                ...
                ```
                where {eigenvalue_i} are eigenvalues assumed to be sorted in
                ascending order.  {eigenvalue_i} and {density_i} are both
                floats.
            '''))

    parser.add_argument(
        '--output_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The output conf file path for generated eigenvalue-dependent
            learning rate schedule configure.
            '''))

    # Parameters related to the scheduling
    parser.add_argument(
        '--min_lr',
        type=float,
		    nargs='?',
        const=None,
		    default=None,
        help=textwrap.dedent(
            '''
            None by default. If specified, the scheduling will do a linear
            scaling for all learning rates to make the learing rate in last
            iteration equal to `min_lr`.
            '''))

    parser.add_argument(
        '--beta', type=float, default=2.0,
        help='The constant which controls the interval for eigenvalues')

    # Parameters related to the optimization problem
    parser.add_argument(
        '--num_iter', type=int, required=True,
        help='The number of iterations')

    # Debug parameters
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def parse_input(input_eigenval_file):
    """Parse eigenvalue distribution from given file.

    Args:
        input_eigenval_file: str, which specifies the path of eigenvalues.
    Returns:
        A numpy with size (D, 2), where D is the number of eigenvalues, where
        each row has two floats (eigenval, density) are the eigenvalue and its
        corresponding density.
    Raises:
        ParseError if eigenvalues are not sorted in ascending order.
    """
    with open(input_eigenval_file, 'r') as fin:
        eigenval_list = []
        density_list = []
        for line in fin:
            eigenval = float(line.split(' ')[0])
            density = float(line.split(' ')[1])
            eigenval_list.append(eigenval)
            density_list.append(density)

    if eigenval_list != sorted(eigenval_list):
        raise ParseError('eigenvalues are not sorted')

    return np.asarray(list(zip(eigenval_list, density_list)))


def get_schedule(eigenval_with_density_np, num_iter, beta):
    """Generates proposed eigenvalue-dependent scheduling.

    Encoded in piecewise inverse time decay scheduling.

    Args:
        eigenval_with_density_np: a 2D numpy of floats with shape (D, 2), where
            D is the number of eigenvalues. Each row contains two floats
            (eigenvalue, density), where eigenvalue is the eigenvalue, density
            is its corresponding density. Rows are sorted in ascending order of
            eigenvalue.
        num_iter: int, number of iterations.
        beta: float, the constant which controls the interval for eigenvalues.

    Returns:
        A tuple of three lists with same length:
            (starting_point_list, a_list, b_list),
        where the schedule will be

            learning_rate = init_lr / (a_i * (t - t_i) + b_i)
                if t in [t_i, t_{i+1}),

            (a_i := a_list[i],
             b_i := b_list[i],
             t_i := starting_point_list[i])
    Raises:
        InvalidArgumentValueError if any provided argument has invalid value.
    """
    # Eigenvalues are guaranteed to be sorted in ascending order
    eigenval_np = eigenval_with_density_np[:, 0]
    mu = eigenval_np[0]
    L = eigenval_np[-1]
    kappa = L / mu
    logging.info('L = %.10lf', L)
    logging.info('mu = %.10lf', mu)

    # Gets density proportion of each interval
    proportion_list = [0]      # Proportion of i-th interval in all iterations
                               #   = Accumulated density in interval
                               #     [t_i, t_{i+1}]
    i = 0
    l_bound = lambda i: mu * (beta ** i)
    r_bound = lambda i: mu * (beta ** (i+1))

    for eigenval, density in eigenval_with_density_np:
        # Finds the interval that "l_bound(i) <= eigenval < r_bound(i)"
        while r_bound(i) <= eigenval:
            i += 1
            proportion_list.append(0)

        proportion_list[i] += density

    # Normalizes proportions by sqrt(s_i) / Z
    proportion_np = np.asarray(proportion_list)
    proportion_np = np.sqrt(proportion_np)
    proportion_np /= proportion_np.sum()
    proportion_list = proportion_np.tolist()

    # Since the iteration number of each interval is an integer instead of a
    # float, we need to take that into account and make the number after taking
    # ceil still sum up to T=num_iter.
    # The strategy here is to reserve 1 iteration for each interval with
    # non-zero proportion, then use floor instead of ceil for each proportion *
    # T, i.e. each interval will have `floor(proportion * T') + 1` iterations,
    # where T' = T - number of intervals with non-zero proprotions.
    num_nonzero_interval = 0
    for proportion in proportion_list:
        if proportion > 0:
            num_nonzero_interval += 1

    num_iter_prime = num_iter - num_nonzero_interval
    if num_iter_prime < 0:
        raise InvalidArgumentValueError(
            'Number of iterations (%d) should be larger than number of'
            'intervals with non-zero proportion (%d)' % (
                num_iter, num_nonzero_interval))

    logging.debug('num_iter = %d', num_iter)
    logging.debug('num_nonzero_interval = %d', num_nonzero_interval)
    logging.debug('num_iter_prime = %d', num_iter_prime)

    # Gets number of interations for each interval
    num_interval = len(proportion_list)
    num_iter_np = np.zeros(num_interval, dtype=int)
    for i, proportion in enumerate(proportion_list):
        if proportion > 0:
            num_iter_np[i] = int(math.floor(num_iter_prime * proportion) + 0.5)
            num_iter_np[i] += 1         # The reserved 1 iteration

    num_iter_remain = num_iter - num_iter_np.sum()
    logging.debug('num_iter_remain = %d', num_iter_remain)

    if num_iter_remain > 0:
        # If there are still some iterations remain, it must <
        # num_nonzero_interval, distribute it one by one for each non-zero
        # interval
        for i, proportion in enumerate(proportion_list):
            if num_iter_remain <= 0:
                break
            if proportion > 0:
                num_iter_np[i] += 1
                num_iter_remain -= 1

    for i, delta in enumerate(num_iter_np):
        logging.debug(
            'Interval %d: eigen value in [%.10lf, %.10lf), proportion = %.10lf'
            ', delta = %d' % (
                i, l_bound(i), r_bound(i), proportion_list[i], num_iter_np[i]))

    # Gets learning rate schedule for each interval.
    # For scaled eigenvalues L' = k * L, mu' = k * mu, for some constant k
    #   eta_t
    #                                  1
    #    =  ---------------------------------------------------------
    #        L' + mu' sum_{j=1}^i d_{j-1} 2^{j-1} + mu' 2^i (t - t_i)
    #
    #                                 1/L'
    #    =  --------------------------------------------------------------
    #        1 + mu'/L' sum_{j=1}^i d_{j-1} 2^{j-1} + mu'/L' 2^i (t - t_i)
    #
    #                                 eta_0
    #    =  ---------------------------------------------------------------
    #       1 + 1/kappa sum_{j=1}^i d_{j-1} 2^{j-1} + 1/kappa 2^i (t - t_i)
    #
    #   for t in [t_i, t_{i+1}), d_i = t_{i+1} - t_i, i >= 0
    #   =>
    #   In piecewise inverse time schedule: lr = eta_0 / [a_i (t - t_i) + b_i]
    #     a_i = 1/kappa * 2^i
    #     b_i = 1 + 1/kappa sum_{j=1}^i d_{j-1} 2^{j-1}
    #         = 1 + 1/kappa sum_{j=1}^{i-1} d_{j-1} 2^{j-1}
    #             + 1/kappa d_{i-1} 2^{i-1}
    #         = b_{i-1} + 1/kappa d_{i-1} 2^{i-1}

    t_list = [0]
    a_list = []
    b_list = []

    for i, delta in enumerate(num_iter_np):
        a = 1 / kappa * (beta ** i)
        b = 1 if i == 0 else b_list[i-1] + 1 / kappa * prev_delta * (beta ** (i-1))
        t = t_list[i] + delta
        prev_delta = delta

        # Updates a_i, b_i, t_{i+1}
        a_list.append(a)
        b_list.append(b)
        t_list.append(t)

    # Removes empty intervals
    final_a_list = []
    final_b_list = []
    final_t_list = []
    for i in range(num_interval):
        if t_list[i] == t_list[i+1]:
            continue
        a = a_list[i]
        b = b_list[i]
        t = t_list[i]
        final_a_list.append(a)
        final_b_list.append(b)
        final_t_list.append(t)

    return final_t_list, final_a_list, final_b_list


def main():
    """Generates configs of eigenvalue-dependent schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Parses eigenvalue distribution from input
    eigenval_with_density_np = parse_input(args.input_eigenval_file)

    # Computes values for the config file
    starting_point_list, a_list, b_list = get_schedule(
        eigenval_with_density_np,
        args.num_iter,
        beta=args.beta)

    # Prepares the config content
    starting_points_str = ', '.join([str(x) for x in starting_point_list])
    a_str = ', '.join([str(a) for a in a_list])
    b_str = ', '.join([str(b) for b in b_list])
    conf_str = textwrap.dedent(
        '''
        [general]
        type = piecewise_inverse_time

        [hyperparams]
        starting_points = {starting_points_str}
        a = {a_str}
        b = {b_str}
        '''.format(
            starting_points_str=starting_points_str,
            a_str=a_str,
            b_str=b_str))

    if args.min_lr is not None:
        conf_str += textwrap.dedent(
            '''
            min_lr = {min_lr}
            num_iter = {num_iter}
            '''.format(
                min_lr=args.min_lr,
                num_iter=args.num_iter))

    # Outputs the config file
    with open(args.output_file, 'w') as fout:
        fout.write(conf_str)
        fout.write('\n')


if __name__ == '__main__':
    main()

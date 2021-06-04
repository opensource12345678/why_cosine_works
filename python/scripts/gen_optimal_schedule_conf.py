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
        description='Generate proposed optimal learning rate schedule conf')

    # Hyperparameters related to the algorithm
    parser.add_argument(
        '--output_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The output conf file path for generated optimal learning rate
            schedule configure.
            '''))

    parser.add_argument(
        '--c', type=float, default=1.0, required=True,
        help=textwrap.dedent(
            '''
            The constant for controlling interval length, specifically T = c
            T', where T is the real iteration number, T' is the iteration
            number used in our analysis. This parameter is useful when the
            trying to strike a balance between SGD variance and its
            corresponding GD bias.
            '''))
    parser.add_argument(
        '--decay_rate', type=float, default=2.0, required=True,
        help=textwrap.dedent(
            '''
            The decay rate in the learning rate schedule, i.e. eta_i = eta_0
            * decay_rate^(-i) for i-th interval, where i starts from 0.
            '''))

    # Parameters related to the optimization problem
    parser.add_argument(
        '--num_iter', type=int, required=True,
        help='The number of iterations')
    parser.add_argument(
        '--init_lr', type=float, required=True,
        help='Initial learning rate')
    parser.add_argument(
        '--lipschitz_const', type=float, required=True,
        help='The Lipschitz smoothness constant L')

    # Debug parameters
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def get_optimal_schedule(c, r, T, L, eta_0):
    """Computes optimal schedule."""
    T /= c

    def get_k(L, eta_0, T):
        r_pow_k_minus_1 = T / ((math.log(2) ** 2) * (1 / (L * eta_0)))
        k_float = math.log(r_pow_k_minus_1) / math.log(r) + 1
        return int(math.ceil(k_float) + 0.5)

    k = get_k(L, eta_0, T)

    def get_ro(eta_i, k, i, T):
        return (1 / eta_i * (k - i) * (math.log(k - i + 1) ** 2)) / float(T)

    def get_t(ro_i, eta_i, t_i):
        return int(math.ceil(1 / (ro_i * eta_i) + t_i) + 0.5)

    def get_eta(eta_0, i):
        return eta_0 * ((1 / float(r)) ** i)

    t_list = [0]
    eta_list = []
    factor_list = []
    ro_list = []

    for i in range(k):
        eta = get_eta(eta_0, i)
        eta_list.append(eta)
        factor_list.append((1 / float(r)) ** i)

        ro = get_ro(eta, k, i, T)
        ro_list.append(ro)

        t = get_t(ro, eta, t_list[i])
        t = int(math.ceil(t_list[i] + c * (t - t_list[i])) + 0.5)
        t_list.append(t)

    starting_point_list = t_list[:-1]
    return starting_point_list, factor_list


def main():
    """Generates configs of proposed near-optimal schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Computes values for the config file
    starting_point_list, factor_list = get_optimal_schedule(
        c=args.c,
        r=args.decay_rate,
        T=args.num_iter,
        L=args.lipschitz_const,
        eta_0=args.init_lr)

    # Prepares the config content
    starting_points_str = ', '.join([str(x) for x in starting_point_list])
    factors_str = ', '.join([str(x) for x in factor_list])
    conf_str = textwrap.dedent(
        '''
        [general]
        type = piecewise_constant

        [hyperparams]
        starting_points = {starting_points_str}
        factors = {factors_str}
        '''.format(
            starting_points_str=starting_points_str,
            factors_str=factors_str))

    # Outputs the config file
    with open(args.output_file, 'w') as fout:
        fout.write(conf_str)
        fout.write('\n')


if __name__ == '__main__':
    main()

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
import sys
import textwrap

import numpy as np

from utils import load_input

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
        description='Calculates hessian matrix for ridge regression')

    parser.add_argument(
        '--input_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The training data file. Each line has a format as follows:
                `{label} 1:{feature_1} 2:{feature_2} ... d:{feature_d}`
            where d is the number of features. For example,
                `15.0 1:-1 2:0.027027 3:0.0420168 4:-0.831858 5:-0.63733`
            This format follows the convention in libsvm dataset. Please refer
            to
                https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html
            for more details.
            '''))

    parser.add_argument(
        '--output_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The output file for eigenvalues. Eigenvalues are sorted in
            ascending order. Each line has a format as follows,
                `{eigenvalue} {1/d}`
            where d is the number of feature dimension.
            '''))

    parser.add_argument(
        '--alpha', type=float, required=True,
        help='The regularization coefficient')
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def get_hess_eigenvalues(feature_matrix, label_array, alpha):
    """Gets eigenvalues for hessian matrix of ridge regression.

    Args:
        feature_matrix: a 2D numpy array of floats with shape N x d.
        label_array: an 1D numpy array of floats with shape N.
        alpha: float, the L2 regularization coefficient.

    Returns:
        A 1D numpy array of floats with shape d, the eigen_values for the
            hessian_matrix in ascending order.
    """
    n, d = feature_matrix.shape
    x = feature_matrix

    # H = 2 X^T X / n + 2 alpha I
    hessian_matrix = 2 * np.matmul(np.transpose(x), x) / n + 2 * alpha * np.identity(d)

    # H = P^T D P, eigenvalues are in ascending order by default
    eigen_values, _ = np.linalg.eigh(hessian_matrix)

    return eigen_values


def main():
    """Uses ridge regression to analyze SGD with learning rate schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Loads input data and converts to numpy
    feature_matrix, label_array = load_input(args.input_file)
    n, d = feature_matrix.shape
    logging.info('load input data complete, %d samples, %d dims', n, d)

    # Calculates eigenvalues of hessian matrix
    alpha = args.alpha
    eigenvalue_np = get_hess_eigenvalues(feature_matrix, label_array, alpha)
    logging.info('compute hessian matrix eigenvalues complete')

    # Outputs eigenvalues
    with open(args.output_file, 'w') as fout:
        for eigenvalue in eigenvalue_np:
            fout.write('%.10lf %.10lf\n' % (eigenvalue, 1 / d))


if __name__ == '__main__':
    main()

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
        description=textwrap.dedent(
            '''
            Preprocess eigenvals, including,

                * Change to absolute value:
                    `lambda <- abs(lambda)`

                * Linear scale:
                    `lambda <- (lambda - mu) / (L - mu) * (L' - mu') + mu'`
                    where L' = new_L, mu' = new_mu
            '''
        ))

    # Arguments related to input/output
    parser.add_argument(
        '--input_file', type=str, required=True,
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
            The output eigenvalue file with same format as 'input_file'.
            '''))

    # Parameters related to preprocessing
    parser.add_argument(
        '--abs', nargs='?', const=True, default=False)

    parser.add_argument(
        '--new_L', type=float, default=None,
        help=textwrap.dedent(
            '''
            The scaled L for scaling eigenvalues. Default: None, which means
            use original L.
            '''))

    parser.add_argument(
        '--new_mu', type=float, default=None,
        help=textwrap.dedent(
            '''
            The scaled mu for scaling eigenvalues. Default: None, which means
            use original mu.
            '''))

    # Debug parameters
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def parse_input(input_file):
    """Parse eigenvalue distribution from given file.

    Args:
        input_file: str, which specifies the path of eigenvalues.
    Returns:
        A numpy with size (D, 2), where D is the number of eigenvalues, where
        each row has two floats (eigenval, density) are the eigenvalue and its
        corresponding density.
    Raises:
        ParseError if eigenvalues are not sorted in ascending order.
    """
    with open(input_file, 'r') as fin:
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


def preprocess(eigenval_with_density_np, args):
    """Preprocesses eigenvalues.

    Args:
        eigenval_with_density_np: a 2D numpy array of floats with shape (D, 2),
            where D is the number of eigenvalues. Each row has two floats
            (eigenval, density). Rows are guaranteed to be sorted in ascending
            order of 'eigenval'.
        args: a struct whose member corresponds to the required (optional)
            variable. Commandline arguments, e.g. args.L = 1.0.
    Returns:
        a numpy array with same shape of 'eigenval_with_density_np', which
        specifies the preprocessed eigenvalues along with their density.
    Raises:
        InvalidArgumentValueError if new_L < new_mu.
    """
    logging.info('----- Transform all eigenvalues to its absolute value...')
    if args.abs:
        eigenval_with_density_np[:, 0] = np.abs(eigenval_with_density_np[:, 0])
        eigenval_np = eigenval_with_density_np[:, 0]
        indices = eigenval_np.argsort()
        eigenval_with_density_np = eigenval_with_density_np[indices]

    logging.info('----- Scale eigenvalues...')
    eigenval_np = eigenval_with_density_np[:, 0]
    mu = eigenval_np[0]
    L = eigenval_np[-1]

    new_mu = args.new_mu if args.new_mu is not None else mu
    new_L = args.new_L if args.new_L is not None else L

    if new_L < new_mu:
        raise InvalidArgumentValueError(
            'Provided new_L (%.10lf) < new_mu (%.10lf)' % (new_L, new_mu))

    for i, (eigenval, _) in enumerate(eigenval_with_density_np):
        new_eigenval = (eigenval - mu) / (L - mu) * (new_L - new_mu) + new_mu
        eigenval_with_density_np[i][0] = new_eigenval

    return eigenval_with_density_np


def output(eigenval_with_density_np, output_file):
    """Output eigenvalues along with their density to output file.

    Args:
        eigenval_with_density_np: a 2D numpy array of floats with shape (D, 2),
            where D is the number of eigenvalues. Each row has two floats
            (eigenval, density). Rows are guaranteed to be sorted in ascending
            order of 'eigenval'.
        output_file: str, which specifies the path of output file.
    """
    with open(output_file, 'w') as fout:
        for eigenval, density in eigenval_with_density_np:
            fout.write('%.100lf %.10lf\n' % (eigenval, density))


def main():
    """Preprocesses eigenvalues"""
    args = parse_argument(sys.argv)

    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    logging.info('===== Parse eigenvalue distribution from input file')
    eigenval_with_density_np = parse_input(args.input_file)

    logging.info('===== Preprocess eigenvalues')
    eigenval_with_density_np = preprocess(eigenval_with_density_np, args)

    logging.info('===== Output eigenvalue to output files')
    output(eigenval_with_density_np, args.output_file)


if __name__ == '__main__':
    main()

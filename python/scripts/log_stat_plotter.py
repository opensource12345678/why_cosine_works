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
import re
import sys
import textwrap

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


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
            Plotting statistics of log files, e.g. training loss, test
            accuracy, learning rate curve.
            '''))

    # Arguments related to input
    parser.add_argument(
        '--input_files', type=str, required=True,
        help=textwrap.dedent(
            '''
            Input log files for specified learning rate scheduling.
            Use ";" to separate multiple files, e.g. "a.log;b.log;c.log".
            Notice that if the number of log files > 7, color duplication may
            occur in the plotted image.
            '''))

    parser.add_argument(
        '--output_img_file_prefix', type=str, required=True,
        help='The prefix for output dir & file path for plotted image')

    # Arguments related to plotting
    parser.add_argument(
        '--num_epoch', type=int, required=True,
        help='The number of epochs')

    parser.add_argument(
        '--batch_size', type=int, required=True,
        help='The size of a batch')

    parser.add_argument(
        '--trainset_size', type=int, required=True,
        help='The number of samples of the training set')

    # Debug parameters
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    parser.add_argument(
        '--output_file_prefix', type=str,
        help=textwrap.dedent(
            '''
            The output file path for debugging. Print statistics during each
            iteration. Row i denotes iteration i-th statistics in the log,
            column j (j >= 2) denotes log file j-2 (also starts from 0) with
            the order in 'log_files'. Each statistics will be outputed to
            `{output_file_prefix}_{stat_name}.txt`, e.g.
                ```
                # output_file_prefix = 'tmp/log-stat-plotter
                tmp/log-stat-plotter_train_loss.txt
                tmp/log-stat-plotter_test_acc.txt
                tmp/log-stat-plotter_learning_rate.txt
                ```

            Output Format:
                ```
                {iter_id_for_this_record} {stat_in_log_0} {stat_in_log_1} ...
                {iter_id_for_this_record} {stat_in_log_0} {stat_in_log_1} ...
                {iter_id_for_this_record} {stat_in_log_0} {stat_in_log_1} ...
                ...
                ```
            '''))

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def parse_stat(input_file_list):
    """Parse statistics, e.g. training loss/test accuracy from input files.

    Args:
        input_file_list: a list of str, which represents the log files.
    Returns:
        A dict maps statistics names to a 2D numpy array, where (i,j) in that
                numpy array means statistics of i-th record for input file j.
                For example,
            ```
            {
                'train_loss': np.asarray([[4.0, 4.2], [3.5, 3.8], [2.1, 3.5]]),
                'test_acc': np.asarray([[0.1, 0.1], [0.2, 0.15], [0.35, 0.2]])
            }
            ```
    Raises:
        ParseError if any error occurs.
    """
    train_loss_list = []        # Element (i,j) has same meaning as output
    test_acc_list = []
    lr_list = []

    train_loss_pattern = re.compile('.*INFO batch loss: (.*)$')
    test_acc_pattern = re.compile('.*test accuracy: (.*)$')
    lr_pattern = re.compile('.*INFO learning rate: (.*)$')

    for file_id, input_file in enumerate(input_file_list):
        train_loss_i = 0
        test_acc_i = 0
        lr_i = 0
        with open(input_file, 'r') as fin:
            for line in fin:
                m = train_loss_pattern.match(line)
                if m and len(m.groups()) == 1:
                    if file_id == 0:
                        train_loss_list.append([])
                    train_loss_list[train_loss_i].append(float(m.group(1)))
                    train_loss_i += 1

                m = test_acc_pattern.match(line)
                if m and len(m.groups()) == 1:
                    if file_id == 0:
                        test_acc_list.append([])
                    test_acc_list[test_acc_i].append(float(m.group(1)))
                    test_acc_i += 1

                m = lr_pattern.match(line)
                if m and len(m.groups()) == 1:
                    if file_id == 0:
                        lr_list.append([])
                    lr_list[lr_i].append(float(m.group(1)))
                    lr_i += 1

    stat_dict = {
        'train_loss': np.asarray(train_loss_list),
        'test_acc': np.asarray(test_acc_list),
        'learning_rate': np.asarray(lr_list)
    }
    return stat_dict


def get_index(stat_dict, num_epoch, batch_size, trainset_size):
    """Prepares indices for statistics.

    This replies on some implicit assumptions about the logging logics of the
    training code.

    Args:
        stat_dict: a dict maps statistics names to a 2D numpy array, where rows
            represents the number of records.
        num_epoch: int.
        batch_size: int.
        trainset_size: int.
    Returns:
        index_dict: a dict maps statistics names to a 1D numpy array, where
            the number of rows is the same as the corresponding numpy array.
            `index_dict[stat_name][i]` means the iteration number for i-th
            record.
    """
    index_dict = {}

    # Training losses and learning rates are printed every
    #   r = (trainset_size//1000 * num_epoch // batch_size) iterations,
    #       i.e. t % r == 0, (t > 0)
    # For CIFAR10/CIFAR100, this is
    #   r = (50 * num_epoch // batch_size) iterations, i.e. t % r == 0, (t > 0)
    r = trainset_size // 1000 * num_epoch // batch_size

    num_record = stat_dict['train_loss'].shape[0]
    index_dict['train_loss'] = np.arange(r, r * (num_record + 1), r) - 1

    num_record = stat_dict['learning_rate'].shape[0]
    index_dict['learning_rate'] = np.arange(r, r * (num_record + 1), r) - 1

    # Test accuracies are printed at the end of each epoch
    r = (trainset_size - 1) // batch_size + 1       # ceiling
    num_record = stat_dict['test_acc'].shape[0]
    index_dict['test_acc'] = np.arange(r, r * (num_record + 1), r) - 1

    return index_dict


def get_nicknames(multi_input_file):
    """Get nicknames for each input log file.

    Args:
        multi_input_file: str. The path(s) of input file(s). If
                there are multiple files, they will be separated by ';',
                e.g. '{input_file_0};{input_file_1};...;{input_file_{n-1}}'.
            Each has a format '{nickname}_...', e.g. 'exp_...', 'inv_...'.
    Returns:
        A list of str, the nicknames for each learning rate schedule, same
        order as in 'multi_input_file'.
    """
    name_list = multi_input_file.split(';')

    # Example: log/xxx/exp_init-lr-0.1.conf -> exp
    nickname_list = [name.split('/')[-1].split('_')[0] for name in name_list]
    return nickname_list


def plot_curve(output_img_file_prefix, stat_name, nickname_list,
               stat_2dnp, index_list_np, num_iter):
    """Exports plotted learning rate curve for given 'lr_list'.

    Args:
        output_img_file_prefix: str, prefix for outputed image path.
        stat_name: str, the name of the statistics.
        nickname_list: a list of str, nickname_list[i] is the nickname for
            i-th input file.
        stat_2dnp: a 2D numpy array, where (i,j) represents statistics of i-th
            record in file j.
        index_list_np: a 1D numpy array, where i-th element is the iteration
            number for i-th record. It has same number of rows as 'stat_2dnp'.
        num_iter: int, number of all iterations.
    """
    num_file = len(nickname_list)
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']     # 7 stadnard colors

    # Original scale
    fig = plt.figure()

    ax = plt.gca()
    ax.set_xlim(0, num_iter)
    if stat_name == 'train_loss':
        ax.set_ylim(0, 1.0)
    elif stat_name == 'test_acc':
        # ax.set_ylim(0.65, 0.75)       # CIFAR100
        ax.set_ylim(0.8, 1.0)         # CIFAR10
    elif stat_name == 'learning_rate':
        ax.set_ylim(0, 0.15)
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('%s' % stat_name)
    ax.grid(axis='x', which='major')
    ax.grid(axis='y', which='major')

    for _ in range(2, 6, 2):
        for file_id in range(num_file):
            plt.plot(
                index_list_np,
                stat_2dnp[:, file_id],
                color_map[file_id % 7],
                label=nickname_list[file_id])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    fig.savefig(output_img_file_prefix + '_' + stat_name + '.png', dpi=fig.dpi)

    # Log scale
    fig = plt.figure()

    ax = plt.gca()
    ax.set_xlim(0, num_iter)
    if stat_name == 'train_loss':
        # ax.set_ylim(1e-8, 1e1)          # No weight decay
        ax.set_ylim(1e-1, 1e1)            # With weight decay
    elif stat_name == 'test_acc':
        ax.set_ylim(1e-1, 1.0)
    elif stat_name == 'learning_rate':
        ax.set_ylim(1e-4, 1.5e-1)
    ax.set_yscale('log')
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('log(%s)' % stat_name)
    ax.grid(axis='x', which='major')
    ax.grid(axis='y', which='major')

    for _ in range(2, 6, 2):
        for file_id in range(num_file):
            plt.plot(
                index_list_np,
                stat_2dnp[:, file_id],
                color_map[file_id % 7],
                label=nickname_list[file_id])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    fig.savefig(output_img_file_prefix + '_' + stat_name + '-log.png',
                dpi=fig.dpi)


def main():
    """Generates configs of proposed near-optimal schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Parses conf file for learning rate schedule
    logging.info('input_files (experiment logs) = %s', args.input_files)
    input_file_list = args.input_files.split(';')

    # Parses statistics from input files
    logging.info('Parse statistics...')
    stat_dict = parse_stat(input_file_list)
    index_dict = get_index(
        stat_dict, args.num_epoch, args.batch_size, args.trainset_size)

    # Outputs this list to "output_file"
    if args.output_file_prefix:
        logging.info('Output files...')
        for stat_name in ['train_loss', 'test_acc', 'learning_rate']:
            output_file = args.output_file_prefix + '_' + stat_name + '.txt'
            with open(output_file, 'w') as fout:
                stat_2dnp = stat_dict[stat_name]
                index_list_np = index_dict[stat_name]
                num_record = stat_2dnp.shape[0]

                for record_id in range(num_record):
                    t = index_list_np[record_id]
                    stat_in_t = stat_2dnp[record_id]
                    stat_in_t_str = ' '.join('%.10lf' % x for x in stat_in_t)

                    # "{t} {input-file_0} {input-file_1} ...
                    fout.write('%d %s\n' % (t, stat_in_t_str))

    # Gets abbreviated name for each scheduler
    nickname_list = get_nicknames(args.input_files)

    # Plots learning rate curve to outputed image file
    logging.info('Plot learning rate...')
    trainset_size = args.trainset_size
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    num_iter = ((trainset_size - 1) // batch_size + 1) * num_epoch

    for stat_name in ['train_loss', 'test_acc', 'learning_rate']:
        plot_curve(args.output_img_file_prefix,
                   stat_name,
                   nickname_list,
                   stat_dict[stat_name],
                   index_dict[stat_name],
                   num_iter)


if __name__ == '__main__':
    main()

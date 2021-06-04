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

from collections import OrderedDict
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np

from utils import parse_lr_schedule_conf


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
        description='Plotting learning curve of a conf')

    # Arguments related to input/output
    parser.add_argument(
        '--lr_schedule_conf_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            Input conf files for specified learning rate scheduling.
            Use ";" to separate multiple files, e.g. "a.conf;b.conf;c.conf".
            Notice that if the number of schedulers > 7, color duplication may
            occur in the plotted image.
            '''))

    parser.add_argument(
        '--output_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The output file path for learning rate during each iteration. Row
            i denotes iteration t=i-1 (starts from 0), column j (j >= 2)
            denotes schedule j-2 (also starts from 0) with the order in
            'lr_schedule_conf_file'.

            Output Format:
                ```
                0 {lr_t=0_for_schedule_0} {lr_t=1_for_scheule_1} ...
                1 {lr_t=1_for_schedule_0} {lr_t=1_for_scheule_1} ...
                2 {lr_t=2_for_schedule_0} {lr_t=2_for_scheule_1} ...
                ...
                ```
            '''))

    parser.add_argument(
        '--output_img_file_prefix', type=str, required=True,
        help='The prefix for output dir & file path for plotted image')

    # Arguments related to plotting
    parser.add_argument(
        '--num_iter', type=int, required=True,
        help='The number of iterations')
    parser.add_argument(
        '--init_lr', type=float, required=True,
        help='Initial learning rate')

    # Debug parameters
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def parse_multiple_lr_schedule_conf(multi_conf_file):
    """Parses learning rate schedules from the given conf file str.

    Args:
        multi_conf_file: str. The path(s) of learning rate schedule(s). If
                there are multiple conf files, they will be separated by ';',
                e.g. '{lr_conf_file_1};{lr_conf_file_2};...;{lr_conf_file_n}'.
    Returns:
        A list of learning rate schedule functions,
		    ```
		    learning_rate = lr_scheduler(init_lr, t)
		    ```
		function arguments:
		    init_lr: float, the intial learning rate;
		    t: int, current iteration number, starts from 0, i.e. we have t=0
		        for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
        UnsupportedSchedulerError if the scheduler type is not found.
	"""
    conf_file_list = multi_conf_file.split(';')
    lr_scheduler_list = []
    for conf_file in conf_file_list:
        lr_schedule_conf = ConfigParser()
        lr_schedule_conf.read(conf_file)
        lr_scheduler = parse_lr_schedule_conf(lr_schedule_conf)
        lr_scheduler_list.append(lr_scheduler)

    return lr_scheduler_list


def get_nicknames(multi_conf_file):
    """Get nicknames for each learning rate scheduler.

    Args:
        multi_conf_file: str. The path(s) of learning rate schedule(s). If
                there are multiple conf files, they will be separated by ';',
                e.g. '{lr_conf_file_1};{lr_conf_file_2};...;{lr_conf_file_n}'.
            Each has a format '{nickname}_...', e.g. 'exp_...', 'inv_...'.
    Returns:
        A list of str, the nicknames for each learning rate schedule, same
        order as in 'multi_conf_file'.
    """
    name_list = multi_conf_file.split(';')

    # Example: log/xxx/exp_init-lr-0.1.conf -> exp
    nickname_list = [name.split('/')[-1].split('_')[0] for name in name_list]
    return nickname_list


def plot_curve(output_img_file_prefix, nickname_list, lr_list):
    """Exports plotted learning rate curve for given 'lr_list'.

    Args:
        output_img_file_prefix: str, prefix for outputed image path.
        nickname_list: a list of str, nickname_list[i] is the nickname for
            i-th learning rate scheduler.
        lr_list: a list of lists, each sublist is a list of floats, and has
            same length. lr_list[i][j] is the learning rate for
                scheduler j in iteration i.
    """
    num_plot = 1000
    num_iter = len(lr_list)
    num_scheduler = len(lr_list[0])

    index = np.arange(0, num_iter, max(1, num_iter // num_plot))
    plot_lr_list = np.asarray(lr_list)[index]

    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']     # 7 stadnard colors

    # Original scale
    fig = plt.figure()

    ax = plt.gca()
    ax.set_xlim(0, index[-1])
    ax.set_ylim(0, 0.15)
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('learning rate')
    ax.grid(axis='x', which='major')
    ax.grid(axis='y', which='major')

    for _ in range(2, 6, 2):
        for sched_id in range(num_scheduler):
            plt.plot(
                index,
                plot_lr_list[:, sched_id],
                color_map[sched_id % 7],
                label=nickname_list[sched_id])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    fig.savefig(output_img_file_prefix + '.png', dpi=fig.dpi)

    # Log scale
    fig = plt.figure()

    ax = plt.gca()
    ax.set_xlim(0, index[-1])
    ax.set_ylim(1e-4, 1.5e-1)
    ax.set_yscale('log')
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('log(learning rate)')
    ax.grid(axis='x', which='major')
    ax.grid(axis='y', which='major')

    for _ in range(2, 6, 2):
        for sched_id in range(num_scheduler):
            plt.plot(
                index,
                plot_lr_list[:, sched_id],
                color_map[sched_id % 7],
                label=nickname_list[sched_id])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    fig.savefig(output_img_file_prefix + '-log.png', dpi=fig.dpi)


def main():
    """Generates configs of proposed near-optimal schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Parses conf file for learning rate schedule
    lr_schedule_conf_file = args.lr_schedule_conf_file
    logging.info('lr_schedule_conf_file = %s', lr_schedule_conf_file)
    lr_scheduler_list = parse_multiple_lr_schedule_conf(lr_schedule_conf_file)

    # Gets learning rates for each iteration
    logging.info('Get learning rate...')
    lr_list = []
    for t in range(args.num_iter):
        # Gets learning rates for each scheduler
        lr_in_t = []
        for lr_scheduler in lr_scheduler_list:
            learning_rate = lr_scheduler(args.init_lr, t)
            lr_in_t.append(learning_rate)
        lr_list.append(lr_in_t)

    # Outputs this list to "output_file"
    with open(args.output_file, 'w') as fout:
        for t, lr_in_t in enumerate(lr_list):
            # "{t} {lr_schedule_0} {lr_schedule_1} ...
            lr_in_t_str = ' '.join(str(lr) for lr in lr_in_t)
            fout.write('%d %s\n' % (t, lr_in_t_str))

    # Gets abbreviated name for each scheduler
    nickname_list = get_nicknames(lr_schedule_conf_file)

    # Plots learning rate curve to outputed image file
    logging.info('Plot learning rate...')
    plot_curve(args.output_img_file_prefix, nickname_list, lr_list)


if __name__ == '__main__':
    main()

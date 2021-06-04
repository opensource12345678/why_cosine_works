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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.googlenet import googlenet
from models.vgg import vgg11_bn
from models.vgg import vgg13_bn
from models.vgg import vgg16_bn
from models.vgg import vgg19_bn

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
        '--weight_decay', type=float, default=0.0,
        help=('Weight decay, also coefficient for L2 term * 2,'
              ' i.e. L2 loss = 1/2 wd * ||w||_2^2'))
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
        '--model', type=str, default='googlenet',
        help=textwrap.dedent(
            '''
            Supported models:
              * googlenet
              * vgg11
              * vgg13
              * vgg16
              * vgg19
            '''))

    # Test parameters
    parser.add_argument(
        '--val', const=True, default=False, nargs='?',
        help='Turn on validation mode')

    # Debug parameters
    parser.add_argument(
        '--pseudo_random', const=True, default=False, nargs='?',
        help='A global option to make all random operations deterministic')
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def load_data(dataset_name, batch_size, validation_mode=False):
    """Loads data in pytorch.

    Args:
        dataset_name: str. Supported datasets ['cifar10', 'cifar100'].
        batch_size: int.
				validation_mode: bool, whether using validation set for test.
    Returns:
        (train_loader, test_loader, stat), where,
            * train_loader: a DataLoader object for loading training data.
            * test_loader, a DataLoader object for loading test data.
            * stat_dict: a dict maps data statistics names to their values, e.g.
                'num_sample', 'num_class'.
    """
    # Chooses dataset
    if dataset_name == 'cifar10':       # CIFAR10
        mean = (0.4914, 0.4822, 0.4465)     # Mean per channel in training set
        stddev = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, stddev)]
        )
        training_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, stddev)]
        )
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        num_class = 10
        image_tensor = training_set[0][0]
        shape = list(image_tensor.size())

    elif dataset_name == 'cifar100':        # CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        stddev = (0.2675, 0.2565, 0.2761)

        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, stddev)]
        )
        training_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, stddev)]
        )
        test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)

        num_class = 100
        image_tensor = training_set[0][0]
        shape = list(image_tensor.size())
    else:
        raise UnsupportedDatasetError('unsupported dataset "%s"' % dataset_name)

    # Prepares dataloader
    num_sample = len(training_set)
    if validation_mode:
      num_train = int(num_sample * 0.9 + 0.5)
      train_indices = np.arange(num_train)
      test_indices = np.arange(num_train, num_sample)
      num_sample = num_train

      train_loader = DataLoader(
          torch.utils.data.Subset(training_set, train_indices),
          shuffle=True,
          num_workers=4,
          batch_size=batch_size)
      test_loader = DataLoader(
          torch.utils.data.Subset(training_set, test_indices),
          shuffle=True,
          num_workers=4,
          batch_size=batch_size)

    else:
      # Normal mode
      train_loader = DataLoader(
          training_set,
          shuffle=True,
          num_workers=4,
          batch_size=batch_size)
      test_loader = DataLoader(
          test_set,
          shuffle=True,
          num_workers=4,
          batch_size=batch_size)

    stat_dict = {}
    stat_dict['num_class'] = num_class
    stat_dict['num_sample'] = num_sample
    stat_dict['num_iter'] = len(train_loader)
    stat_dict['shape'] = shape

    return train_loader, test_loader, stat_dict

def optimize_in_pytorch(train_loader, test_loader, num_class,
                        lr_scheduler, args, model_type):
    """Runs optimization in tensorflow.

    Args:
        train_loader: DataLoader for loading batches of training samples.
        test_loader: DataLoader for loading batches of test samples.
        num_class: int, the number of classes.

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
        model_type: a str specifies the model type. Supported types are:
            'resnet18'

    Raises:
        UnsupportedModelError if 'model_type' not supported.
    """
    # Chooses model
    if model_type == 'googlenet':
        model = googlenet(num_class=num_class)
    elif model_type == 'vgg11':
        model = vgg11_bn(num_class=num_class)
    elif model_type == 'vgg13':
        model = vgg13_bn(num_class=num_class)
    elif model_type == 'vgg16':
        model = vgg16_bn(num_class=num_class)
    elif model_type == 'vgg19':
        model = vgg19_bn(num_class=num_class)
    else:
        raise UnsupportedModelError('unsupported model type "%s"' % model_type)

    model = model.cuda()

    # Prepares for training
    optimizer = optim.SGD(
        model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    mean_crossent_loss = nn.CrossEntropyLoss()

    # Training
    def train_on_batch(x_batch, y_batch):
        optimizer.zero_grad()
        predicted_logits = model(x_batch)
        loss = mean_crossent_loss(predicted_logits, y_batch)
        loss.backward()
        optimizer.step()

        # Computes the implicit l2 loss brought by weight_decay
        with torch.no_grad():
            l2_coeff = args.weight_decay * 0.5
            l2_reg = torch.tensor(0.).cuda()
            for param in model.parameters():
                if param.requires_grad:             # Trainable parameters only
                    l2_reg += torch.norm(param) ** 2
            l2_loss = l2_coeff * l2_reg
        return loss.item(), loss.item() + l2_loss.item()

    @torch.no_grad()
    def test_on(data_loader):
        model.eval()
        sum_test_loss = 0.0
        num_correct = 0.0
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            predicted_logits = model(x_batch)
            loss = mean_crossent_loss(predicted_logits, y_batch)
            sum_test_loss += loss.item()
            _, predicted_ys = predicted_logits.max(1)       # max over dim 1
            num_correct += predicted_ys.eq(y_batch).sum()

        num_batch = len(data_loader.dataset)
        test_loss = sum_test_loss / float(num_batch)
        test_acc = num_correct.float() / float(num_batch)

        return test_loss, test_acc

    t = 0
    train_loss = 0.0
    train_loss_with_l2 = 0.0
    for epoch_id in range(args.num_epoch):
        logging.info('========== epoch %d ==========', epoch_id)
        model.train()
        for x_batch, y_batch in train_loader:

            # Updates learning rate
            learning_rate = lr_scheduler(args.init_lr, t)
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
            t += 1

            # Training on a single batch
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            batch_loss, batch_loss_with_l2 = train_on_batch(x_batch, y_batch)
            train_loss += batch_loss
            train_loss_with_l2 += batch_loss_with_l2

            # if t % (50 * args.num_epoch // args.batch_size) == 0:
            if True:
                logging.info('learning rate: %.10lf', learning_rate)
                logging.info('batch loss: %.10lf', batch_loss)
                logging.info('batch loss with l2: %.10lf', batch_loss_with_l2)

        num_batch = len(train_loader)
        train_loss /= float(num_batch)
        train_loss_with_l2 /= float(num_batch)

        # Tests at the end of every epoch
        test_loss, test_acc = test_on(test_loader)

        # Prints summarized statistics at the end of every epoch
        logging.info(
            'End of epoch %d'
            ', crossentropy loss only: %.10lf'
            ', moving-average train loss: %.10lf'
            ', test loss: %.10lf, test accuracy: %.6lf',
            epoch_id,
            train_loss,
            train_loss_with_l2,
            test_loss,
            test_acc)

        train_loss = 0.0
        train_loss_with_l2 = 0.0

    # Gets training loss for final result. Notice it is not the moving loss
    # anymore, but the real loss over the whole dataset.
    logging.info('========== summary ==========')
    train_loss, train_acc = test_on(train_loader)

    logging.info('Summary: train loss: %.10lf, train accuracy: %.6lf',
                 train_loss, train_acc)


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
        torch.manual_seed(0)
        torch.set_deterministic(True)

    # Loads training/test data
    train_loader, test_loader, stat_dict = load_data(args.dataset,
                                                     args.batch_size,
                                                     args.val)

    logging.info('image shape = %s', str(stat_dict['shape']))
    logging.info('number of iterations = %s', str(stat_dict['num_iter']))
    logging.info('load data complete, %d samples', stat_dict['num_sample'])

    # Runs optimization
    optimize_in_pytorch(
        train_loader,
        test_loader,
        stat_dict['num_class'],
        lr_scheduler,
        args,
        model_type=args.model)


if __name__ == '__main__':
    main()

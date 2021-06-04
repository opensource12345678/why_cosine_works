#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from density_plot import get_esd_plot
from models.googlenet import googlenet
from models.torchvision_resnet import resnet18 as torchvision_resnet18
from models.vgg import vgg16_bn
from models.resnet import resnet
from pyhessian import hessian

# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')
parser.add_argument('--q', type=int)
parser.add_argument('--model-name',
                    type=str,
                    default='resnet')
parser.add_argument('--dataset',
                    type=str,
                    default='cifar10')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, test_loader, num_class = getData(name=args.dataset,
                                               train_bs=args.mini_hessian_batch_size,
                                               test_bs=1)
##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (50000 % args.hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

# get model
model_name = args.model_name
if model_name == 'resnet':
    model = resnet(num_classes=num_class,
				   depth=20,
				   residual_not=args.residual,
				   batch_norm_not=args.batch_norm)
elif model_name == 'torchvision_resnet':
    print('num_class = %d' % num_class)
    model = torchvision_resnet18(num_classes=num_class)
elif model_name == 'googlenet':
    model = googlenet()
elif model_name == 'vgg16':
    model = vgg16_bn()

if args.cuda:
    model = model.cuda()
model = torch.nn.DataParallel(model)

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

count_parameters(model)

criterion = nn.CrossEntropyLoss()  # label loss

###################
# Get model checkpoint, get saving folder
###################
if args.resume == '':
    raise Exception("please choose the trained model")
model.load_state_dict(torch.load(args.resume))

######################################################
# Begin the computation
######################################################

# turn model to eval mode
model.eval()
if batch_num == 1:
    hessian_comp = hessian(model,
                           criterion,
                           data=hessian_dataloader,
                           cuda=args.cuda)
else:
    hessian_comp = hessian(model,
                           criterion,
                           dataloader=hessian_dataloader,
                           cuda=args.cuda)

print(
    '********** finish data londing and begin Hessian computation **********')

top_eigenvalues, _ = hessian_comp.eigenvalues()
trace = hessian_comp.trace()
density_eigen, density_weight = hessian_comp.density(iter=args.q)

print('\n***Top Eigenvalues: ', top_eigenvalues)
print('\n***Trace: ', np.mean(trace))

print('\n**len(density_eigen): ', len(density_eigen))
print('\n**len(density_weight): ', len(density_weight))
print('\n**len(density_eigen[0]): ', len(density_eigen[0]))
print('\n**len(density_weight[0]): ', len(density_weight[0]))
print('sum(density_weight[0]): ', sum(density_weight[0]))

output_file = 'output/eigenvalue_model-%s_iter-%d.txt' % (model_name, args.q)
with open(output_file, 'w') as fout:
    n_v = len(density_eigen)
    for i in range(n_v):
        for eigen, weight in zip(density_eigen[i], density_weight[i]):
            fout.write('%.10lf %.10lf\n' % (eigen, weight))

get_esd_plot(density_eigen, density_weight, args.q, model_name)

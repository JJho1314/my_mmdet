from mmdet.models.backbones.clip_image import clip_image
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer

import ipdb
import clip
from mmcv.runner import BaseModule

from torch.nn import functional as F

from collections import OrderedDict
from typing import Tuple, Union

import logging
import torch
import torch.distributed as dist

def print_parameter_grad_info(net):
    print('-------parameters requires grad info--------')
    for name, p in net.named_parameters():
        print(f'{name}:\t{p.requires_grad}')

def print_net_state_dict(net):
    for key, v in net.state_dict().items():
        print(f'{key}')

if __name__ == "__main__":
    net = clip_image().cuda().float()

    print_parameter_grad_info(net)
    net.requires_grad_(False)
    print_parameter_grad_info(net)

    torch.random.manual_seed(5)
    test_data = torch.rand(1, 3, 32, 32).cuda()
    train_data = torch.rand(5, 3, 32, 32).cuda()

    # print(test_data)
    # print(train_data[0, ...])
    for epoch in range(3):
        # training phase, 假设每个epoch只迭代一次
        net.train()
        pre = net(train_data)
        
        net.eval()
        x = net(test_data)
        print(f'epoch:{epoch}', x)


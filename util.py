"""Utility functions."""

import os
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')

def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

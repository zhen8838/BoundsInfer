import math
import torch
import numpy as np


param_in_hws = [
    [3, 4],
    [32, 24]
]
param_in_channels = [
    1,
    2,
    16,
]
param_output_channels = [
    2,
    1,
    8,
]
param_k_hws = [
    [3, 3],
    # [1,1]
]
param_strides = [
    [1, 1],
    # [2,2]
]
param_paddings = [
    [0, 0],
    # [1,1]
]
param_output_paddings = [
    [0, 0],
]
param_dilations = [
    [1, 1]
]


def conv2d_transpose_reference(input: np.ndarray, weight: np.ndarray, stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1]):
  # in_channels = input.shape[1]
  # kernel_size = [weight.shape[2], weight.shape[3]]
  # OC = weight.shape[1]  # note groups == 1
  return torch.conv_transpose2d(torch.tensor(input), torch.tensor(weight), None, stride, padding, output_padding, 1, dilation).detach().numpy()


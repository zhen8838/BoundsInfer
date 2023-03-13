import math
import torch
import numpy as np


param_in_hws = [
    [24, 24]
]
param_in_channels = [
    1,
    16,
]
param_output_channels = [
    1,
    32,
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


def calc_input_kw_index_by_kernel_index(out_index: int, padding_before: int, padding_after: int, input_length: int, kernel_index: int, stride: int):
  in_index = max(0, int(math.ceil(1 * (out_index - kernel_index) / stride)))
  in_index -= padding_before
  in_index = math.max(0, math.min(in_index, input_length - 1))
  return in_index


def calc_input_kh_range(out_index: int, padding_before: int, padding_after: int, input_length: int, kernel_length: int, stride: int):
  in_index_start = calc_input_kw_index_by_kernel_index(
      out_index, padding_before, padding_after, input_length, 0, stride)
  in_index_end = calc_input_kw_index_by_kernel_index(
      out_index, padding_before, padding_after, input_length, kernel_length - 1, stride)

  return slice(in_index_start, in_index_end)

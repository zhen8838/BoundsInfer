from torch import nn
import typing
import torch
import numpy as np
from typing import Tuple


def get_windowed_output_size(size: int, filter: int, stride: int, dilation: int, padding: np.ndarray):
  effective_filter_size = (filter - 1) * dilation + 1
  return int((size + padding[0] + padding[1] - effective_filter_size + stride) / stride)


def gnne_pdp_reduce(input: np.ndarray, output: np.ndarray,
                    filter: Tuple[int, int],
                    stride: Tuple[int, int], padding: np.ndarray):
  stride_h, stride_w = (stride[0], stride[1])
  filter_h, filter_w = (filter[0], filter[1])
  padding_h = (padding[0], padding[0])
  padding_w = (padding[1], padding[1])
  in_shape = input.shape
  out_h = get_windowed_output_size(in_shape[2], filter_h, stride_h, 1, padding_h)
  out_w = get_windowed_output_size(in_shape[3], filter_w, stride_w, 1, padding_w)
  out_shape = [in_shape[0], in_shape[1], out_h, out_w]

  for batch in range(in_shape[0]):
    for oc in range(in_shape[1]):
      for oy in range(out_h):
        for ox in range(out_w):
          in_y_origin = (oy * stride_h) - padding_h[0]
          in_x_origin = (ox * stride_w) - padding_w[0]

          # filter_y_start = std::max(0, -in_y_origin);
          # filter_y_end = std::min(filter_h, ()in_shape[2] - in_y_origin);
          # filter_x_start = std::max(0, -in_x_origin);
          # filter_x_end = std::min(filter_w, ()in_shape[3] - in_x_origin);
          # float value = input[linear_index(in_shape, { batch, oc, (size_t)(in_y_origin + filter_y_start), (size_t)((in_x_origin + filter_x_start)) })];


def rerange(value: np.ndarray, rg):
  scale = (rg[1] - rg[0]) / value.size
  return value * scale + rg[0]


def test_reducewindow_boundinfer():
  input = (rerange(np.arange(1 * 3 * 64 * 64), (0, 100))).reshape(1, 3, 64, 64)

  filter = (3, 3)
  stride = (1, 2)
  padding = (0, 0)
  excepted = nn.AvgPool2d(kernel_size=filter, stride=stride,
                          padding=padding)(torch.from_numpy(input))

  output = np.zeros_like(excepted)
  gnne_pdp_reduce(input, output, filter=filter, stride=stride, padding=padding)

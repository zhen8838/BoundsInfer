import torch
import pytest
from typing import Tuple
import numpy as np

param_in_hws = [
    [224, 244]
]
param_in_channels = [
    3
]
param_output_channels = [
    32
]
param_k_hws = [
    [3, 3]
]
param_strides = [
    [2, 2]
]
param_paddings = [
    [[0, 1],
     [0, 1]]
]
param_dilations = [
    [1, 1]
]
param_groups = [
    1
]


def run_forward(input: np.ndarray, weight: np.ndarray, in_hw, in_channel, output_channel, k_hw, stride, padding, dilation, group):
  paded = np.pad(input, [[0, 0], [0, 0], padding[0], padding[1]])
  return torch.conv2d(torch.tensor(paded), torch.tensor(weight), None, stride, 0, dilation, group).detach().numpy()


def get_windowed_output_size(size: int, filter: int,
                             stride: int, dilation: int,
                             padding: Tuple[int, int]):
  effective_filter_size = (filter - 1) * dilation + 1
  return int((int(size) + padding[0] + padding[1] - effective_filter_size + stride) // stride)


def get_output_shape(out_channels, in_hw, k_hw, stride, dilation, padding):
  out_h = get_windowed_output_size(in_hw[0], k_hw[0], stride[0], dilation[0], padding[0])
  out_w = get_windowed_output_size(in_hw[1], k_hw[1], stride[1], dilation[1], padding[1])
  return [1, out_channels, out_h, out_w]


def run_backward_v1(input: np.ndarray, weight: np.ndarray, in_hw, in_channel, output_channel, k_hw, stride, padding, dilation, group):
  in_shape = input.shape
  output_shape = get_output_shape(output_channel, in_hw, k_hw, stride, dilation, padding)
  test_output = np.zeros(output_shape, np.float32)
  out_groups = output_channel // group
  in_groups = in_channel // group
  for ob in range(output_shape[0]):
    for oc in range(output_shape[1]):
      for oh in range(output_shape[2]):
        for ow in range(output_shape[3]):
          in_y_origin = (oh * stride[0]) - padding[0][0]
          in_x_origin = (ow * stride[1]) - padding[1][0]
          filter_y_start = int(max(0, (-in_y_origin + dilation[0] - 1) / dilation[0]))
          filter_y_end = int(
              min(k_hw[0], (in_shape[2] - in_y_origin + dilation[0] - 1) / dilation[0]))
          filter_x_start = int(max(0, (-in_x_origin + dilation[1] - 1) / dilation[1]))
          filter_x_end = int(
              min(k_hw[1], (in_shape[3] - in_x_origin + dilation[1] - 1) / dilation[1]))

          in_y_start = in_y_origin + dilation[0] * filter_y_start
          in_y_end = in_y_origin + dilation[0] * filter_y_end

          in_x_start = in_x_origin + dilation[1] * filter_x_start
          in_x_end = in_x_origin + dilation[1] * filter_x_end

          in_c_start = (oc // out_groups) * in_groups
          in_c_end = in_c_start + in_groups

          test_output[ob, oc, oh, ow] = (
              # conv2d.bias[oc:oc + 1] +
              (input[ob:ob + 1, in_c_start:in_c_end,
                     in_y_start: in_y_end:dilation[0],
                     in_x_start: in_x_end:dilation[1]] *
               weight[oc:oc + 1, 0: in_groups, filter_y_start:filter_y_end, filter_x_start:filter_x_end]).sum())
  return test_output


@pytest.mark.parametrize('in_hw', param_in_hws)
@pytest.mark.parametrize('in_channel', param_in_channels)
@pytest.mark.parametrize('output_channel', param_output_channels)
@pytest.mark.parametrize('k_hw', param_k_hws)
@pytest.mark.parametrize('stride', param_strides)
@pytest.mark.parametrize('padding', param_paddings)
@pytest.mark.parametrize('dilation', param_dilations)
@pytest.mark.parametrize('group', param_groups)
def test_forward_backward(in_hw, in_channel, output_channel, k_hw, stride, padding, dilation, group):
  weight = np.random.randn(output_channel, in_channel, k_hw[0], k_hw[1]).astype(
      np.float32)  # np.ones(kw, np.float32)
  input = np.random.randn(1, in_channel, in_hw[0], in_hw[1]).astype(
      np.float32)  # np.ones(iw).astype(np.float32)
  output_forward = run_forward(input, weight, in_hw, in_channel,
                               output_channel, k_hw, stride, padding, dilation, group)
  output_backward_v1 = run_backward_v1(
      input, weight, in_hw, in_channel, output_channel, k_hw, stride, padding, dilation, group)
  assert np.allclose(output_forward, output_backward_v1, atol=1e-5)


if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

# import halide as hl
import torch
import numpy as np
# from halide import BoundaryConditions
from typing import NamedTuple, Tuple
import pytest
from tests.conv2d_transpose_utils import *


def naive_impl_v2(input: np.ndarray, weight: np.ndarray, stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1]):
  """ 
  naive v2, calc index by output index
  """
  B = input.shape[0]
  OC, IC = (weight.shape[1], weight.shape[0])
  OH = (input.shape[2] - 1) * stride[0] - 2 * padding[0] + \
      dilation[0] * (weight.shape[2] - 1) + output_padding[0] + 1
  OW = (input.shape[3] - 1) * stride[1] - 2 * padding[1] + \
      dilation[1] * (weight.shape[3] - 1) + output_padding[1] + 1
  out_shape = [B, OC, OH, OW]
  output = np.zeros(out_shape)
  for batch in range(input.shape[0]):
    for oc in range(OC):
      # print("====================")
      for oh in range(OH):
        for ow in range(OW):
          # print(f"output[{oc},{oh},{ow}] += ", end="")
          iy_origin = (oh + padding[0]) // stride[0]
          ix_origin = (ow + padding[1]) // stride[1]
          for ic in range(IC):
            for ky in range(weight.shape[2]):
              for kx in range(weight.shape[3]):
                iy = iy_origin - ky
                ix = ix_origin - kx
                if ((iy >= 0 and iy <= input.shape[2] - 1) and
                        (ix >= 0 and ix <= input.shape[3] - 1)):
                  # print(f"input[{ic}, {iy}, {ix}] * weight[{oc}, {ic}, {ky}, {kx}]", end=", ")
                  output[batch, oc, oh, ow] += input[batch, ic, iy, ix] * weight[ic, oc, ky, kx]
          # print()
  return output


@pytest.mark.parametrize('in_hw', param_in_hws)
@pytest.mark.parametrize('in_channel', param_in_channels)
@pytest.mark.parametrize('output_channel', param_output_channels)
@pytest.mark.parametrize('k_hw', param_k_hws)
@pytest.mark.parametrize('stride', param_strides)
@pytest.mark.parametrize('padding', param_paddings)
@pytest.mark.parametrize('output_padding', param_output_paddings)
@pytest.mark.parametrize('dilation', param_dilations)
def test_torch_naive_v2(in_hw: Tuple[int, int], in_channel: int, output_channel: int, k_hw: Tuple[int, int],
                        stride: Tuple[int, int], padding: Tuple[int, int], output_padding: Tuple[int, int], dilation: Tuple[int, int]):
  input_shape = [1, in_channel, in_hw[0], in_hw[1]]
  weight_shape = [in_channel, output_channel, k_hw[0], k_hw[1]]
  input = np.random.randn(*input_shape).astype(np.float32)
  weight = np.random.randn(*weight_shape).astype(np.float32)

  output_torch = conv2d_transpose_reference(
      input, weight, stride, padding, output_padding, dilation)
  output_naive = naive_impl_v2(input, weight, stride, padding, output_padding, dilation)

  assert (np.allclose(output_torch, output_naive, atol=1e-5))


if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

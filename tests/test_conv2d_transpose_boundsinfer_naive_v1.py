import numpy as np
from typing import NamedTuple, Tuple
import pytest
from .conv2d_transpose_utils import *


def naive_impl_v1(input: np.ndarray, weight: np.ndarray, stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1]):
  """ 
  weight [ic,oc/groups,kh,kw]
  input [b,ic,ih,iw]
  output [b,oc,oh,ow]
  oh = (ih-1)*stirde_h - 2*pad_h + dilation_h * (kh - 1) + output_padding_h + 1
  ow = (iw-1)*stirde_w - 2*pad_w + dilation_w * (kw - 1) + output_padding_w + 1
  """
  B = input.shape[0]
  OC = weight.shape[1]
  OH = (input.shape[2] - 1) * stride[0] - 2 * padding[0] + \
      dilation[0] * (weight.shape[2] - 1) + output_padding[0] + 1
  OW = (input.shape[3] - 1) * stride[1] - 2 * padding[1] + \
      dilation[1] * (weight.shape[3] - 1) + output_padding[1] + 1
  out_shape = [B, OC, OH, OW]
  output = np.zeros(out_shape)
  for batch in range(input.shape[0]):
    for oc in range(OC):
      for ic in range(input.shape[1]):
        for iy in range(input.shape[2]):
          for ix in range(input.shape[3]):
            out_y_origin = (iy * stride[0]) - padding[0]  # 这里不用before了 .before;
            out_x_origin = (ix * stride[1]) - padding[1]  # 这里不用before了 .before;
            filter_y_start = max(0, (-out_y_origin + dilation[0] - 1) / dilation[0])
            filter_y_end = min(weight.shape[2], (int(out_shape[2]) - out_y_origin + dilation[0] - 1) / dilation[0])
            filter_x_start = max(0, (-out_x_origin + dilation[1] - 1) / dilation[1])
            filter_x_end = min(weight.shape[3], (int(out_shape[3]) - out_x_origin + dilation[1] - 1) / dilation[1])
            for ky in range(filter_y_start, filter_y_end):
              for kx in range(filter_x_start, filter_x_end):
                out_y = out_y_origin + dilation[0] * ky
                out_x = out_x_origin + dilation[1] * kx
                output[batch, oc, out_y, out_x] += input[batch, ic, iy, ix] * weight[ic, oc, ky, kx]
  return output


@pytest.mark.parametrize('in_hw', param_in_hws)
@pytest.mark.parametrize('in_channel', param_in_channels)
@pytest.mark.parametrize('output_channel', param_output_channels)
@pytest.mark.parametrize('k_hw', param_k_hws)
@pytest.mark.parametrize('stride', param_strides)
@pytest.mark.parametrize('padding', param_paddings)
@pytest.mark.parametrize('output_padding', param_output_paddings)
@pytest.mark.parametrize('dilation', param_dilations)
def test_torch_naive_v1(in_hw: Tuple[int, int], in_channel: int, output_channel: int, k_hw: Tuple[int, int],
                        stride: Tuple[int, int], padding: Tuple[int, int], output_padding: Tuple[int, int], dilation: Tuple[int, int]):
  input_shape = [1, in_channel, in_hw[0], in_hw[1]]
  weight_shape = [in_channel, output_channel, k_hw[0], k_hw[1]]
  input = np.random.randn(*input_shape).astype(np.float32)
  weight = np.random.randn(*weight_shape).astype(np.float32)

  output_torch = conv2d_transpose_reference(
      input, weight, stride, padding, output_padding, dilation)
  output_naive = naive_impl_v1(input, weight, stride, padding, output_padding, dilation)

  assert (np.allclose(output_torch, output_naive, atol=1e-5))


if __name__ == "__main__":
  pytest.main(['-vv', __file__,'-s'])

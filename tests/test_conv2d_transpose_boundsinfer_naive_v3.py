from src import Segments, Conv2dTransposeBoundsInfer
import torch
import numpy as np
# from halide import BoundaryConditions
from typing import NamedTuple, Tuple
import pytest
from tests.conv2d_transpose_utils import *


@pytest.mark.parametrize('in_hw', param_in_hws)
@pytest.mark.parametrize('in_channel', param_in_channels)
@pytest.mark.parametrize('output_channel', param_output_channels)
@pytest.mark.parametrize('k_hw', param_k_hws)
@pytest.mark.parametrize('stride', param_strides)
@pytest.mark.parametrize('padding', param_paddings)
@pytest.mark.parametrize('output_padding', param_output_paddings)
@pytest.mark.parametrize('dilation', param_dilations)
def test_torch_naive_v3(in_hw: Tuple[int, int], in_channel: int, output_channel: int, k_hw: Tuple[int, int],
                        stride: Tuple[int, int], padding: Tuple[int, int], output_padding: Tuple[int, int], dilation: Tuple[int, int]):
  input_shape = [1, in_channel, in_hw[0], in_hw[1]]
  weight_shape = [in_channel, output_channel, k_hw[0], k_hw[1]]
  infer = Conv2dTransposeBoundsInfer(in_channel, output_channel, k_hw, 1, False,
                                     padding, output_padding, stride, dilation, input_shape)

  input = infer.test_input.detach().numpy()
  weight = infer.w.detach().numpy()  # [ic, oc, kh, kw]
  target_output = infer.target_output.detach().numpy()
  test_output = np.zeros_like(target_output)

  for ob in Segments(0, infer.output_shape[0], 1):
    for oc in Segments(0, infer.output_shape[1], 1):
      for oh in Segments(0, infer.output_shape[2], 1):
        for ow in Segments(0, infer.output_shape[3], 1):
          input_tile = input[infer.get_input_segment(ob, oc, oh, ow)]
          w_tile = weight[infer.get_w_segment(oc, oh, ow)][:, :, ::-1, ::-1]
          output_tile = test_output[ob, oc, oh, ow]
          # note w is [ic,oc,kh,kw], input is [b,ic,kh,kw], so we need transpose w
          output_tile += np.sum(input_tile * np.transpose(w_tile, [1, 0, 2, 3]), keepdims=True)

  assert (np.allclose(target_output, test_output, atol=1e-5))


if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

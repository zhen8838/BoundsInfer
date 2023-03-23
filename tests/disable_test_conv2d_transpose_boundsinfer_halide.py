import halide as hl
import numpy as np
from halide import BoundaryConditions
from typing import NamedTuple, Tuple
import pytest
from tests.conv2d_transpose_utils import *


def halide_impl(input: np.ndarray, weight: np.ndarray, stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1]):
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
  output = np.zeros(out_shape).astype(np.float32)
  output = output.transpose(list(range(len(output.shape) - 1, -1, -1)))

  hlInput = hl.Buffer(input.transpose(list(range(len(input.shape) - 1, -1, -1))))
  hlWeight = hl.Buffer(weight.transpose(list(range(len(weight.shape) - 1, -1, -1))))
  # hlOutput = hl.Buffer(output.transpose(list(range(len(output.shape) - 1, -1, -1))))
  Batch, Oc, Oh, Ow = hl.Var("Batch"), hl.Var("Oc"), hl.Var("Oh"), hl.Var("Ow")
  r = hl.RDom([(0, weight.shape[3]), (0, weight.shape[2]), (0, IC)], name="r")
  conved = hl.Func("conved")
  hlOutput = hl.Func("output")
  hlOutputBuffer = hl.Buffer(output)
  paded = BoundaryConditions.constant_exterior(
      hlInput, 0.0, [(0, input.shape[-1]), (0, input.shape[-2])])
  conved[Ow, Oh, Oc, Batch] = 0.0
  conved[Ow, Oh, Oc, Batch] += paded[((Ow + padding[1]) / stride[1]) - r.x,
                                     ((Oh + padding[0]) / stride[0]) - r.y, r.z, Batch] * hlWeight[r.x, r.y, Oc, r.z]
  hlOutput[Ow, Oh, Oc, Batch] = conved[Ow, Oh, Oc, Batch]
  OcO, OcI = hl.Var("OcO"), hl.Var("OcI")
  IcO, IcI = hl.RVar("IcO"), hl.RVar("IcI")
  conved.reorder(Oc, Ow, Oh, Batch)
  conved.update().reorder(r.x, r.y, r.z, Oc, Ow, Oh, Batch). \
      split(r.z, IcO, IcI, 8). \
      split(Oc, OcO, OcI, 24, hl.TailStrategy.GuardWithIf). \
      reorder(r.x, r.y, IcI, OcI, IcO, OcO, Ow, Oh, Batch)
  hlOutput.reorder(Oc, Ow, Oh, Batch)
  hlOutput.print_loop_nest()
  hlOutput.realize(hlOutputBuffer)
  output = output.transpose(list(range(len(output.shape) - 1, -1, -1)))
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
  output_naive = halide_impl(input, weight, stride, padding, output_padding, dilation)

  assert (np.allclose(output_torch, output_naive, atol=1e-5))


if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

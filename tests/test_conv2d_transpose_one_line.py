import pytest
import numpy as np


def compute_ow(IW: int, padding: int, stride: int, dilation: int, KW: int):
  return (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1


def run_forward(IW: int, input: np.ndarray, weight: np.ndarray, padding: int, stride: int, dilation: int, KW: int):
  OW = (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1
  output = np.zeros(OW)
  for ix in range(IW):
    out_x_origin = (ix * stride) - padding  # 这里不用before了 .before;
    filter_x_start = max(0, (-out_x_origin + dilation - 1) // dilation)
    filter_x_end = min(KW, (int(OW) -
                            out_x_origin + dilation - 1) // dilation)
    for kx in range(filter_x_start, filter_x_end):
      out_x = out_x_origin + dilation * kx
      # print(out_x, ix)
      output[out_x] += input[ix] * weight[kx]
  return output


def __get_input_range(ox: int, IW: int, padding: int, stride: int, dilation: int, KW: int) -> slice:
  ix_origin = (ox + padding) // stride
  ix_start = ix_origin - (KW - 1)
  ix_end = ix_origin - (0)
  return slice(max(0, ix_start), min(IW - 1, ix_end) + 1)


def get_input_segment(o_seg: slice, IW: int, padding: int, stride: int, dilation: int, KW: int):
  p0 = __get_input_range(o_seg.start, IW, padding, stride, dilation, KW)
  p1 = __get_input_range(o_seg.stop - 1, IW, padding, stride, dilation, KW)
  return slice(p0.start, p1.stop)


def __get_w_range(ox: int, IW: int, padding: int, stride: int, dilation: int, KW: int) -> slice:
  ix_origin = (ox + padding) // stride
  ix_start = ix_origin - (KW - 1)
  ix_end = ix_origin - (0)

  left_overflow = max(0, 0 - ix_start)
  kx_end = KW - left_overflow

  right_overflow = max(0, ix_end - (IW - 1))
  kx_start = 0 + right_overflow
  return slice(kx_start, kx_end)


def get_w_segment(o_seg: slice, IW: int, padding: int, stride: int, dilation: int, KW: int):
  p0 = __get_w_range(o_seg.start, IW, padding, stride, dilation, KW)
  p1 = __get_w_range(o_seg.stop - 1, IW, padding, stride, dilation, KW)
  return slice(p0.start, p1.stop)


def run_backward_v1(IW: int, input: np.ndarray, weight: np.ndarray, padding: int, stride: int, dilation: int, KW: int):
  OW = (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1
  output = np.zeros(OW)
  for ox in range(OW):
    # print(ox, " : ")
    ix_origin = (ox + padding) // stride
    # kernel在ix上向左滑动
    for kx in range(KW):
      ix = ix_origin - kx
      if (ix >= 0 and ix <= IW - 1):
        # print(f'{input[ix]} * {weight[kx]}', end=', ')
        output[ox] += input[ix] * weight[kx]
    # print()
  return output


def run_backward_v2(IW: int, input: np.ndarray, weight: np.ndarray, padding: int, stride: int, dilation: int, KW: int):
  OW = (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1
  output = np.zeros(OW)
  for ox in range(0, OW, 1):
    # print(ox, " : ")
    o_seg = slice(ox, ox + 1)
    input_tile = input[get_input_segment(o_seg, IW, padding, stride, dilation, KW)]
    # 因为kernel在input上反向滑动, 所以这里需要倒着取值.
    weight_tile = weight[get_w_segment(o_seg, IW, padding, stride, dilation, KW)][::-1]
    # print(f'{input_tile} * {weight_tile}')
    output[o_seg] += np.sum(input_tile * weight_tile, keepdims=True)
  return output


def test_forward_backward():
  padding = 0
  stride = 1
  dilation = 1  # not support dilation > 1
  KW = 3
  IW = 24
  weight = np.random.randn(KW).astype(np.float32)
  input = np.arange(IW).astype(np.float32)
  output_forward = run_forward(IW, input, weight, padding, stride, dilation, KW)
  output_backward_v1 = run_backward_v1(IW, input, weight, padding, stride, dilation, KW)
  output_backward_v2 = run_backward_v2(IW, input, weight, padding, stride, dilation, KW)
  assert (np.allclose(output_forward, output_backward_v1))
  assert (np.allclose(output_forward, output_backward_v2))


if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

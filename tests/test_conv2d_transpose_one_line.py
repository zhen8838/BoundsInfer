import pytest
import math
import numpy as np


def compute_ow(IW: int, padding: int, stride: int, dilation: int, KW: int):
  return (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1


def run_forward(IW: int, input: np.ndarray, weight: np.ndarray, padding: int, stride: int, dilation: int, KW: int):
  OW = (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1
  output = np.zeros(OW)
  print("forward :")
  for ix in range(IW):
    out_x_origin = (ix * stride) - padding  # 这里不用before了;
    filter_x_start = max(0, (-out_x_origin + dilation - 1) // dilation)
    filter_x_end = min(KW, (int(OW) - out_x_origin + dilation - 1) // dilation)
    for kx in range(filter_x_start, filter_x_end):
      out_x = out_x_origin + dilation * kx
      # print(out_x, ix)
      output[out_x] += input[ix] * weight[kx]
      print(f"output[{out_x}] += input[{ix}] * weight[{kx}]")
  return output


def run_backward_v1(IW: int, input: np.ndarray, weight: np.ndarray, padding: int, stride: int, dilation: int, KW: int):
  OW = (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1
  output = np.zeros(OW)
  print("backward_v1 :")
  for ox in range(OW):
    # 计算当前ox落在哪个ix的映射domain中.
    (ix_origin, kx_offset) = divmod((ox + padding), stride)
    # 计算当前的kernel offset下可以向左滑动几次.
    kernel_sliding = int(math.ceil((KW - kx_offset) / stride))
    # 在if不够时无法向左滑动
    sliding_stop = min(ix_origin + 1, kernel_sliding)
    # ix_origin超过IW的时候, 相当于已经向左滑动过kernel了, 所以更新ix的起点
    sliding_start = max(0, ix_origin - (IW - 1))
    for s in range(sliding_start, sliding_stop):
      kx = kx_offset + s * stride
      ix = ix_origin - s
      output[ox] += input[ix] * weight[kx]
      print(f"output[{ox}] += input[{ix}] * weight[{kx}]")
  return output


def __get_input_range(ox: int, IW: int, padding: int, stride: int, dilation: int, KW: int) -> slice:
  (ix_origin, kx_offset) = divmod((ox + padding), stride)
  kernel_sliding = int(math.ceil((KW - kx_offset) / stride))
  sliding_stop = min(ix_origin + 1, kernel_sliding)
  sliding_start = max(0, ix_origin - (IW - 1))
  return slice(ix_origin - sliding_stop + 1, ix_origin - sliding_start + 1)


def __get_w_range(ox: int, IW: int, padding: int, stride: int, dilation: int, KW: int) -> slice:
  (ix_origin, kx_offset) = divmod((ox + padding), stride)
  kernel_sliding = int(math.ceil((KW - kx_offset) / stride))
  sliding_stop = min(ix_origin + 1, kernel_sliding)
  sliding_start = max(0, ix_origin - (IW - 1))
  return slice(kx_offset + sliding_start * stride,
               kx_offset + sliding_stop * stride,
               stride)


def get_input_segment(o_seg: slice, IW: int, padding: int, stride: int, dilation: int, KW: int):
  p0 = __get_input_range(o_seg.start, IW, padding, stride, dilation, KW)
  p1 = __get_input_range(o_seg.stop - 1, IW, padding, stride, dilation, KW)
  assert p0.step == p1.step
  return slice(p0.start, p1.stop, p0.step)


def get_w_segment(o_seg: slice, IW: int, padding: int, stride: int, dilation: int, KW: int):
  p0 = __get_w_range(o_seg.start, IW, padding, stride, dilation, KW)
  p1 = __get_w_range(o_seg.stop - 1, IW, padding, stride, dilation, KW)
  assert p0.step == p1.step
  return slice(p0.start, p1.stop, p0.step)


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


IWs = [5, 6, 14]
KWs = [3, 1]
paddings = [2, 1]  # , 1, 0
strides = [1, 2]  # 1


@pytest.mark.parametrize('iw', IWs)
@pytest.mark.parametrize('kw', KWs)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('stride', strides)
def test_forward_backward(iw, kw, padding, stride):
  dilation = 1  # not support dilation > 1
  weight = np.random.rand(kw).astype(np.float32)  # np.ones(kw, np.float32)
  input = np.random.rand(iw).astype(np.float32)  # np.ones(iw).astype(np.float32)
  output_forward = run_forward(iw, input, weight, padding, stride, dilation, kw)
  output_backward_v1 = run_backward_v1(iw, input, weight, padding, stride, dilation, kw)
  output_backward_v2 = run_backward_v2(iw, input, weight, padding, stride, dilation, kw)
  assert (np.allclose(output_forward, output_backward_v1))
  assert (np.allclose(output_forward, output_backward_v2))


if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

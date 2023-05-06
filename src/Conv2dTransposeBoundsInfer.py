from torch import nn
from typing import List, Tuple
import torch


class Conv2dTransposeBoundsInfer():
  def __init__(self, in_channels=16, out_channels=10, kernel_size=3, groups=2, bias=True,
               padding=(1, 2), output_padding=(0, 0), stride=(2, 3), dilation=(1, 1), intput_shape=(3, 16, 24, 24)) -> None:
    self.conv2dtrans = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    self.w = self.conv2dtrans.weight  # [oc, ic/groups, kh, kw]
    self.test_input: torch.Tensor = torch.rand(intput_shape)
    self.in_shape = self.test_input.shape
    self.target_output: torch.Tensor = self.conv2dtrans(self.test_input)
    self.test_output = self.target_output.clone().fill_(0)
    self.output_shape = self.target_output.shape
    self.out_groups = self.conv2dtrans.out_channels // groups
    self.in_groups = self.conv2dtrans.in_channels // groups
    self.dilation_h = self.conv2dtrans.dilation[0]
    self.dilation_w = self.conv2dtrans.dilation[1]
    self.stride_h = self.conv2dtrans.stride[0]
    self.stride_w = self.conv2dtrans.stride[1]
    self.padding_h = self.conv2dtrans.padding[0]
    self.padding_w = self.conv2dtrans.padding[1]
    self.output_padding_h = self.conv2dtrans.output_padding[0]
    self.output_padding_w = self.conv2dtrans.output_padding[1]
    self.filter_h = self.w.shape[2]
    self.filter_w = self.w.shape[3]

  def __get_input_range(self, ob: int, oc: int, oh: int, ow: int) -> List[slice]:

    ix_origin = (ow + self.padding_w) // self.stride_w
    ix_start = ix_origin - (self.filter_w - 1)
    ix_end = ix_origin - (0)
    x_slice = slice(max(0, ix_start), min(self.in_shape[3] - 1, ix_end) + 1)

    iy_origin = (oh + self.padding_h) // self.stride_h
    iy_start = iy_origin - (self.filter_h - 1)
    iy_end = iy_origin - (0)
    y_slice = slice(max(0, iy_start), min(self.in_shape[2] - 1, iy_end) + 1)

    in_c_start = (oc // self.out_groups) * self.in_groups
    in_c_end = in_c_start + self.in_groups
    c_slice = slice(in_c_start, in_c_end)

    return [slice(ob, ob + 1), c_slice, y_slice, x_slice]

  def __get_w_range(self, oc: slice, oh: slice, ow: slice) -> List[slice]:
    ix_origin = (ow + self.padding_w) // self.stride_w
    ix_start = ix_origin - (self.filter_w - 1)
    ix_end = ix_origin - (0)
    ix_left_overflow = max(0, 0 - ix_start)
    kx_end = self.filter_w - ix_left_overflow
    ix_right_overflow = max(0, ix_end - (self.in_shape[3] - 1))
    kx_start = 0 + ix_right_overflow
    kx_slice = slice(kx_start, kx_end)

    iy_origin = (oh + self.padding_h) // self.stride_h
    iy_start = iy_origin - (self.filter_h - 1)
    iy_end = iy_origin - (0)
    iy_left_overflow = max(0, 0 - iy_start)
    ky_end = self.filter_h - iy_left_overflow
    iy_right_overflow = max(0, iy_end - (self.in_shape[2] - 1))
    ky_start = 0 + iy_right_overflow
    ky_slice = slice(ky_start, ky_end)
    # w: [ic,oc,kh,kw]

    in_c_start = (oc // self.out_groups) * self.in_groups
    in_c_end = in_c_start + self.in_groups
    in_c_slice = slice(in_c_start, in_c_end)

    in_c_start = oc % self.out_groups
    in_c_end = in_c_start + 1
    o_c_slice = slice(in_c_start,in_c_end)

    return [in_c_slice, o_c_slice, ky_slice, kx_slice]

  def get_input_segment(self, n: slice, c: slice, h: slice, w: slice) -> List[slice]:
    return [slice(p[0].start, p[1].stop) for p in zip(self.__get_input_range(n.start, c.start, h.start, w.start),
                                                      self.__get_input_range(n.stop - 1, c.stop - 1, h.stop - 1, w.stop - 1))]

  def get_w_segment(self, oc: slice, oh: slice, ow: slice) -> List[slice]:
    return [slice(p[0].start, p[1].stop) for p in zip(self.__get_w_range(oc.start, oh.start, ow.start),
                                                      self.__get_w_range(oc.stop - 1, oh.stop - 1, ow.stop - 1))]

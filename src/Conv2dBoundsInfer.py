from torch import nn
from typing import List, Tuple
import torch


class Conv2dBoundsInfer():
  def __init__(self, in_channels=16, out_channels=10, kernel_size=3, groups=2, bias=True,
               padding=(1, 2), stride=(2, 3), dilation=(1, 1), intput_shape=(3, 16, 24, 24),test = True) -> None:
    self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, dilation, groups, bias)

    self.w = self.conv2d.weight  # [oc, ic/groups, kh, kw]
    self.test_input: torch.Tensor = torch.rand(intput_shape)
    self.in_shape = self.test_input.shape
    self.target_output: torch.Tensor = self.conv2d(self.test_input)
    self.test_output = self.target_output.clone().fill_(0)
    self.output_shape = self.target_output.shape
    self.out_groups = self.conv2d.out_channels // groups
    self.in_groups = self.conv2d.in_channels // groups
    self.dilation_h = self.conv2d.dilation[0]
    self.dilation_w = self.conv2d.dilation[1]
    self.stride_h = self.conv2d.stride[0]
    self.stride_w = self.conv2d.stride[1]
    self.padding_h = self.conv2d.padding[0]
    self.padding_w = self.conv2d.padding[1]
    self.filter_h = self.w.shape[2]
    self.filter_w = self.w.shape[3]
    if test:
      for ob in range(self.output_shape[0]):
        for oc in range(self.output_shape[1]):
          for oh in range(self.output_shape[2]):
            for ow in range(self.output_shape[3]):
              in_y_origin = (oh * self.stride_h) - self.padding_h
              in_x_origin = (ow * self.stride_w) - self.padding_w
              filter_y_start = int(max(0, (-in_y_origin + self.dilation_h - 1) / self.dilation_h))
              filter_y_end = int(
                  min(self.filter_h, (self.in_shape[2] - in_y_origin + self.dilation_h - 1) / self.dilation_h))
              filter_x_start = int(max(0, (-in_x_origin + self.dilation_w - 1) / self.dilation_w))
              filter_x_end = int(
                  min(self.filter_w, (self.in_shape[3] - in_x_origin + self.dilation_w - 1) / self.dilation_w))

              in_y_start = in_y_origin + self.dilation_h * filter_y_start
              in_y_end = in_y_origin + self.dilation_h * filter_y_end

              in_x_start = in_x_origin + self.dilation_w * filter_x_start
              in_x_end = in_x_origin + self.dilation_w * filter_x_end

              in_c_start = (oc // self.out_groups) * self.in_groups
              in_c_end = in_c_start + self.in_groups

              self.test_output[ob, oc, oh, ow] = (self.conv2d.bias[oc:oc + 1] +
                                                  (self.test_input[ob:ob + 1, in_c_start:in_c_end,
                                                                  in_y_start: in_y_end:self.dilation_h,
                                                                  in_x_start: in_x_end:self.dilation_w] *
                                                  self.w[oc:oc + 1, 0: self.in_groups, filter_y_start:filter_y_end, filter_x_start:filter_x_end]).sum())

      # print((self.target_output - self.test_output).max())
      assert (torch.allclose(self.target_output, self.test_output, atol=1e-5))

  def __get_input_range(self, ob, oc, oh, ow) -> List[slice]:
    in_y_origin = (oh * self.stride_h) - self.padding_h
    in_x_origin = (ow * self.stride_w) - self.padding_w
    filter_y_start = int(max(0, (-in_y_origin + self.dilation_h - 1) / self.dilation_h))
    filter_y_end = int(
        min(self.filter_h, (self.in_shape[2] - in_y_origin + self.dilation_h - 1) / self.dilation_h))
    filter_x_start = int(max(0, (-in_x_origin + self.dilation_w - 1) / self.dilation_w))
    filter_x_end = int(
        min(self.filter_w, (self.in_shape[3] - in_x_origin + self.dilation_w - 1) / self.dilation_w))

    in_y_start = in_y_origin + self.dilation_h * filter_y_start
    in_y_end = in_y_origin + self.dilation_h * filter_y_end

    in_x_start = in_x_origin + self.dilation_w * filter_x_start
    in_x_end = in_x_origin + self.dilation_w * filter_x_end

    in_c_start = (oc // self.out_groups) * self.in_groups
    in_c_end = in_c_start + self.in_groups

    return [slice(ob, ob + 1), slice(in_c_start, in_c_end), slice(in_y_origin, in_y_origin + self.dilation_h * self.filter_h), slice(in_x_origin, in_x_origin + self.dilation_w * self.filter_w)]

  def __get_w_range(self, oc) -> List[slice]:
    return [slice(oc, oc + 1), slice(0, self.in_groups), slice(0, self.filter_h), slice(0, self.filter_w)]

  def get_input_segment(self, n: slice, c: slice, h: slice, w: slice) -> List[slice]:
    return [slice(p[0].start, p[1].stop) for p in zip(self.__get_input_range(n.start, c.start, h.start, w.start),
                                                      self.__get_input_range(n.stop - 1, c.stop - 1, h.stop - 1, w.stop - 1))]

  def get_w_segment(self, c: slice) -> List[slice]:
    return [slice(p[0].start, p[1].stop) for p in zip(self.__get_w_range(c.start),
                                                      self.__get_w_range(c.stop - 1))]


def Segments(start: int, stop: int, step: int = 1) -> slice:
  for i in range(start, stop, step):
    yield slice(i, i + step)

from torch import nn
import torch

def test_conv2d_groups():
  groups = 2  # group 指的是把输入channel 和输出channel都分这么多组. 然后每一组内部还是和普通卷积一样的.
  conv2d = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, groups=2, bias=True,
                    padding=(1, 1), stride=(1, 1))


  w = conv2d.weight  # [oc, ic/groups, kh, kw]

  input = torch.rand(3, 16, 24, 24)
  in_shape = input.shape
  target_output: torch.Tensor = conv2d(input)
  test_output = target_output.clone().fill_(0)
  output_shape = target_output.shape
  out_groups = conv2d.out_channels // groups
  in_groups = conv2d.in_channels // groups

  for ob in range(output_shape[0]):
    for oc in range(output_shape[1]):
      for oh in range(output_shape[2]):
        for ow in range(output_shape[3]):
          in_y_origin = (oh * conv2d.stride[0]) - conv2d.padding[0]
          in_x_origin = (ow * conv2d.stride[1]) - conv2d.padding[1]
          # NOTE 这里其实不应该去限制. 应该是对于input需要限制
          filter_y_start = int(max(0, (-in_y_origin + conv2d.dilation[0] - 1) / conv2d.dilation[0]))
          filter_y_end = int(min(
              conv2d.kernel_size[0], (in_shape[2] - in_y_origin + conv2d.dilation[0] - 1) / conv2d.dilation[0]))
          filter_x_start = int(max(0, (-in_x_origin + conv2d.dilation[1] - 1) / conv2d.dilation[1]))
          filter_x_end = int(
              min(conv2d.kernel_size[1], (in_shape[3] - in_x_origin + conv2d.dilation[1] - 1) / conv2d.dilation[1]))

          in_y_start = in_y_origin + conv2d.dilation[0] * filter_y_start
          in_y_end = in_y_origin + conv2d.dilation[0] * filter_y_end

          in_x_start = in_x_origin + conv2d.dilation[1] * filter_x_start
          in_x_end = in_x_origin + conv2d.dilation[1] * filter_x_end

          in_c_start = (oc // out_groups) * in_groups
          in_c_end = in_c_start + in_groups

          test_output[ob, oc, oh, ow] =\
              (conv2d.bias[oc:oc + 1] +
              (input[ob:ob + 1, in_c_start:in_c_end,
                      in_y_start: in_y_end,
                      in_x_start: in_x_end] *
                  w[oc:oc + 1, 0: in_groups, filter_y_start:filter_y_end, filter_x_start:filter_x_end]).sum())


  print((target_output - test_output).max())
  assert(torch.allclose(target_output, test_output, atol=1e-6))


  # def get_input_range(ob, oc, oh, ow):
  #   in_y_origin = (oh * conv2d.stride[0]) - conv2d.padding[0]
  #   in_x_origin = (ow * conv2d.stride[1]) - conv2d.padding[1]
  #   filter_y_start = int(max(0, (-in_y_origin + conv2d.dilation[0] - 1) / conv2d.dilation[0]))
  #   filter_y_end = int(min(
  #       conv2d.kernel_size[0], (in_shape[2] - in_y_origin + conv2d.dilation[0] - 1) / conv2d.dilation[0]))
  #   filter_x_start = int(max(0, (-in_x_origin + conv2d.dilation[1] - 1) / conv2d.dilation[1]))
  #   filter_x_end = int(min(
  #       conv2d.kernel_size[1], (in_shape[3] - in_x_origin + conv2d.dilation[1] - 1) / conv2d.dilation[1]))

  #   print("w:", (oc, oc + 1), (0, in_groups), (filter_y_start,
  #         filter_y_end), (filter_x_start, filter_x_end))

  #   in_c_start = (oc // out_groups) * in_groups
  #   in_c_end = in_c_start + in_groups
  #   return (ob, ob + 1),\
  #       (in_c_start, in_c_end), \
  #       (in_y_origin + conv2d.dilation[0] * filter_y_start,
  #        in_y_origin + conv2d.dilation[0] * filter_y_end),\
  #       (in_x_origin + conv2d.dilation[1] * filter_x_start,
  #        in_x_origin + conv2d.dilation[1] * filter_x_end)


  # def get_input_segment(n, c, h, w):
  #   print([(p[0][0], p[1][1]) for p in zip(get_input_range(
  #       n[0], c[0], h[0], w[0]), get_input_range(n[1], c[1], h[1], w[1]))])


  # get_input_segment((1, 2), (3, 8), (3, 4), (2, 5))

  # get_input_range(0, 0, 0, 5)

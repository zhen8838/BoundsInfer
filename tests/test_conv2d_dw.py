import torch
import torch.nn.functional as f
from functools import partial
import torch.nn as nn
import pytest
import numpy as np


def make_feat(shape: list, default=-1):
  n = 1
  for s in shape:
    n *= s
  arr = torch.arange(0, n, 1)
  if default != -1:
    arr = arr.fill_(default)
  return arr.reshape(*shape)


def do_conv(width=3,
            height=3,
            channel=1,
            paddingX=0,
            paddingY=0,
            strideX=1,
            strideY=1,
            dilationX=1,
            dilationY=1,
            kSizeX=3,
            kSizeY=3,
            filters=1,
            feat=None,
            kern=None):
  feat = make_feat((1, channel, height, width))
  kern = make_feat((channel, 1, kSizeY, kSizeX))
  print(feat)
  print(kern)
  # b c h w,这里的padding是倒着生效的。
  feat = f.pad(feat, (paddingX, paddingX, paddingY, paddingY))
  # print(feat, kern)
  out = f.conv2d(feat, kern, stride=[strideY, strideX],
                 dilation=[dilationY, dilationX], groups=channel)

  print()
  print(out)


i_channel = [3]
i_size = [[6, 6]]
padding = [[0, 0]]
stride = [[1, 1]]
dilation = [[1, 1]]
kernel = [[1, 1]]
o_channel = [1]


@pytest.mark.parametrize('i_channel', i_channel)
@pytest.mark.parametrize('i_size', i_size)
@pytest.mark.parametrize('padding', padding)
@pytest.mark.parametrize('stride', stride)
@pytest.mark.parametrize('dilation', dilation)
@pytest.mark.parametrize('kernel', kernel)
@pytest.mark.parametrize('o_channel', o_channel)
def test_conv_dw(i_channel,
                 i_size,
                 padding,
                 stride,
                 dilation,
                 kernel,
                 o_channel):
  channel = i_channel
  height = i_size[0]
  width = i_size[1]
  paddingY = padding[0]
  paddingX = padding[1]
  strideY = stride[0]
  strideX = stride[1]
  dilationY = dilation[0]
  dilationX = dilation[1]
  kSizeY = kernel[0]
  kSizeX = kernel[1]
  filters = o_channel
  do_conv(width, height, channel, paddingX, paddingY, strideX,
          strideY, dilationX, dilationY, kSizeX, kSizeY, filters)

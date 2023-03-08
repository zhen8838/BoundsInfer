import torch
import torch.nn.functional as f
from functools import partial
import torch.nn as nn
import pytest
import numpy as np


def test_focus_to_conv2d():
  dwkern = np.array([[[1, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [1, 0]],
                    [[0, 0], [0, 1]]]).astype(np.float32)

  convkern = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]).reshape(3, 3, 1, 1).astype(np.float32)

  feat = np.random.rand(1, 3, 224, 224).astype(np.float32)


  def get_one_dwconv_out(i):
    one_kern = dwkern[None, i:i + 1, :, :]
    one_kern = np.repeat(one_kern, 3, 0)
    out = f.conv2d(torch.from_numpy(feat),
                  torch.from_numpy(one_kern),
                  stride=[2, 2],
                  dilation=[1, 1],
                  groups=3)
    return out.numpy()


  def get_one_conv_out():
    return f.conv2d(torch.from_numpy(feat),
                    torch.from_numpy(convkern),
                    stride=[2, 2],
                    dilation=[1, 1],
                    groups=1).numpy()


  dwconvout = np.concatenate([get_one_dwconv_out(i) for i in range(4)], 1)
  dwconvout.shape


  patch_top_left = feat[..., ::2, ::2]
  patch_top_right = feat[..., ::2, 1::2]
  patch_bot_left = feat[..., 1::2, ::2]
  patch_bot_right = feat[..., 1::2, 1::2]
  sliceout = np.concatenate(
      (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), 1)

  print(np.allclose(dwconvout[0, 0:3], patch_top_left))
  print(np.allclose(dwconvout[0, 3:6], patch_top_right))
  print(np.allclose(dwconvout[0, 6:9], patch_bot_left))
  print(np.allclose(dwconvout[0, 9:12], patch_bot_right))


  # 前面加conv1x1 stride=1 可以解决这top left和top right的问题
  convout = f.conv2d(torch.from_numpy(feat),
                    torch.from_numpy(convkern),
                    stride=[1, 1],
                    dilation=[1, 1],
                    groups=1).numpy()
  print(np.allclose(convout[..., ::2, ::2], patch_top_left))
  print(np.allclose(convout[..., ::2, 1::2], patch_top_right))

  # bot left被改成 stride=[2,2] padding[1,0] 然后slice [1: , :]的
  convout_bot_left = f.conv2d(torch.from_numpy(feat),
                              torch.from_numpy(convkern),
                              stride=[2, 2],
                              dilation=[1, 1],
                              padding=[1, 0],
                              groups=1).numpy()
  print(np.allclose(convout_bot_left[..., 1:, :], patch_bot_left))

  # bot right被改成 stride=[2,2] padding[1,1] 然后slice [1: , 1:]的
  convout_bot_right = f.conv2d(torch.from_numpy(feat),
                              torch.from_numpy(convkern),
                              stride=[2, 2],
                              dilation=[1, 1],
                              padding=[1, 1],
                              groups=1).numpy()
  print(np.allclose(convout_bot_right[..., 1:, 1:], patch_bot_right))

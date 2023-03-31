import torch
import pytest

def test_conv2d_transpose_pad():
  w = torch.randn(1, 1, 3, 3, dtype=torch.float32)
  input = torch.ones(1, 1, 4, 4, dtype=torch.float32)
  output = torch.conv_transpose2d(input, w, stride=2, padding=[0, 0])
  output_padding = torch.conv_transpose2d(input, w, stride=2, padding=[2, 2])
  # print(output)
  # print(output_padding)
  # print(output[:, :, 2:-2, 2:-2])
  assert torch.allclose(output_padding, output[:, :, 2:-2, 2:-2])

def test_conv2d_transpose_stride():
  w = torch.ones(1, 1, 1, 3, dtype=torch.float32)
  input = torch.ones(1, 1, 1, 5, dtype=torch.float32)
  output = torch.conv_transpose2d(input, w, stride=2, padding=[0, 0])
  print(output)



if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])
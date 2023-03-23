import halide as hl
import numpy as np
from halide import BoundaryConditions
from typing import NamedTuple, Tuple
import pytest
from tests.conv2d_transpose_utils import *


def halide_impl(input: np.ndarray, weight: np.ndarray, stride=[1, 1], padding=[0, 0], dilation=[1, 1]):
  input = hl.Buffer(np.ones([100, 100], np.float32))
  weights = hl.Buffer(np.ones([3, 3], np.float32))
  outputBuffer = hl.Buffer(hl.Float(32), [100, 100])
  output = hl.Func("output")  # hl.Buffer(hl.Float(32), [100, 100])
  conv = hl.Func("conv")
  win = hl.RDom([(0, 3), (0, 3)])
  x, y = hl.Var("x"), hl.Var("y")
  xo, yo = hl.Var("xo"), hl.Var("yo")
  xi, yi = hl.Var("xi"), hl.Var("yi")
  conv[x, y] = 0.0
  paded = BoundaryConditions.constant_exterior(input, 0.0, [(0, 100), (0, 100)])
  conv[x, y] = conv[x, y] + paded[x + win.x, y + win.y] * weights[win.x, win.y]
  output[x, y] = conv[x, y]
  conv.update(0).tile(x, y, xo, yo, xi, yi, 10, 10)

  output.print_loop_nest()

  output.compile_to_lowered_stmt("1.html", [], hl.StmtOutputFormat.HTML)


if __name__ == "__main__":
  halide_impl(1, 2, 3, 4, 5)

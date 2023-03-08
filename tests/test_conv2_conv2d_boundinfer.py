from src import Conv2dBoundsInfer

# for h in range(0, 26, 12):
#   print((h, min(h + 12, 26)))
#   mid_seg = infer2.get_input_segment((0, 3), (0, 5), (h, min(h + 12, 26)), (0, 28))
#   in_seg = infer1.get_input_segment(*mid_seg)


def test_get_sg():
  infer1 = Conv2dBoundsInfer(in_channels=16, out_channels=10, kernel_size=3, groups=1, bias=True,
                             padding=(1, 2), stride=(1, 1), dilation=(1, 1), intput_shape=(3, 16, 24, 24))
  infer1.output_shape  # [3,10,24,26]

  infer2 = Conv2dBoundsInfer(in_channels=10, out_channels=5, kernel_size=3, groups=1, bias=True,
                             padding=(2, 2), stride=(1, 1), dilation=(1, 1), intput_shape=infer1.output_shape)
  print(infer2.output_shape)  # [3, 5, 12, 13]

  def get_sg(tp):
    mid_seg = infer2.get_input_segment(slice(0, 3), slice(0, 5), tp, slice(0, 28))
    in_seg = infer1.get_input_segment(*mid_seg)

  get_sg(slice(12, 20))
  get_sg(slice(20, 24))

  get_sg(slice(24, 26))

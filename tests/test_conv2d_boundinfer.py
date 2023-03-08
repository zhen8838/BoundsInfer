


# infer = Conv2dBoundsInfer(in_channels=6, out_channels=10, kernel_size=3, groups=2, bias=True,
#                           padding=(0, 0), stride=(1, 1), dilation=(1, 1), intput_shape=(1, 6, 6, 6))

# infer.get_input_segment((0, 1), (0, 1), (0, 2), (0, 2))


# infer.get_input_segment((0, 1), (0, 6), (1, 3), (3, 4))

# infer = Conv2dBoundsInfer(in_channels=16, out_channels=10, kernel_size=3, groups=2, bias=True,
#                           padding=(1, 2), stride=(2, 3), dilation=(1, 1), intput_shape=(3, 16, 24, 24))

# # infer.get_input_segment((1,2), (3,8), (3,4), (2,5))
# infer.get_input_segment((0,1), (0,5), (0,1), (8,9))

# infer.get_input_segment((0, 1), (0, 6), (1, 3), (3, 4))

# infer = Conv2dBoundsInfer(in_channels=16, out_channels=10, kernel_size=3, groups=1, bias=True,
#                           padding=(1, 2), stride=(1, 1), dilation=(1, 1), intput_shape=(3, 16, 24, 24))
# print( infer.output_shape )
# infer.get_input_segment((1,2), (3,8), (3,4), (2,5))
# infer.get_input_segment((0,1), (0,5), (0,1), (8,9))

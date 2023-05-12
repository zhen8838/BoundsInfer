import numpy as np
import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime as ort


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def tanh(x):
  return np.tanh(x)


def run_reference(x: np.ndarray, w: np.ndarray, r: np.ndarray, direction, hidden_size, seq_length, batch_size, input_size, bias: np.ndarray, sequence_lens, initial_h: np.ndarray, initial_c: np.ndarray, output_y: bool, output_h: bool, output_c: bool):
  nodes_inputs = []
  nodes_outputs = []
  initializers = []
  attributes_dict = {}
  nodes = []
  graph_inputs = []
  graph_outputs = []

  num_directions = 2 if direction == 'bidirectional' else 1
  if direction is not None:
    attributes_dict['direction'] = direction
  attributes_dict['hidden_size'] = hidden_size

  # input
  input_shape = [seq_length, batch_size, input_size]
  input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
  nodes_inputs.append('input')
  graph_inputs.append(input)

  w_shape = [num_directions, 4 * hidden_size, input_size]
  w_tensor = helper.make_tensor(
      'W',
      TensorProto.FLOAT,
      dims=w_shape,
      vals=w.flatten().tolist()
  )
  nodes_inputs.append('W')
  initializers.append(w_tensor)

  r_shape = [num_directions, 4 * hidden_size, hidden_size]
  r_tensor = helper.make_tensor(
      'R',
      TensorProto.FLOAT,
      dims=r_shape,
      vals=r.flatten().tolist()
  )
  nodes_inputs.append('R')
  initializers.append(r_tensor)

  # bias
  if bias is None:
    nodes_inputs.append('')
  else:
    bias_shape = [num_directions, 8 * hidden_size]
    bias_tensor = helper.make_tensor(
        'B',
        TensorProto.FLOAT,
        dims=bias_shape,
        vals=bias.flatten().tolist()
    )
    nodes_inputs.append('B')
    initializers.append(bias_tensor)

  if sequence_lens is None:
    nodes_inputs.append('')
  else:
    sequence_lens_shape = [batch_size]
    sequence_lens_tensor = helper.make_tensor(
        'sequence_lens',
        TensorProto.INT32,
        dims=sequence_lens_shape,
        vals=np.full(sequence_lens_shape, seq_length).flatten().tolist()
    )
    nodes_inputs.append('sequence_lens')
    initializers.append(sequence_lens_tensor)

  if initial_h is None:
    nodes_inputs.append('')
  else:
    initial_h_shape = [num_directions, batch_size, hidden_size]
    initial_h_tensor = helper.make_tensor(
        'initial_h',
        TensorProto.FLOAT,
        dims=initial_h_shape,
        vals=initial_h.flatten().tolist()
    )
    nodes_inputs.append('initial_h')
    initializers.append(initial_h_tensor)

  if initial_c is None:
    nodes_inputs.append('')
  else:
    initial_c_shape = [num_directions, batch_size, hidden_size]
    initial_c_tensor = helper.make_tensor(
        'initial_c',
        TensorProto.FLOAT,
        dims=initial_c_shape,
        vals=initial_c.flatten().tolist()
    )
    nodes_inputs.append('initial_c')
    initializers.append(initial_c_tensor)

  # output
  if output_y is False:
    nodes_outputs.append('')
  else:
    output_shape = [seq_length, num_directions, batch_size, hidden_size]
    output = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)
    nodes_outputs.append('Y')
    graph_outputs.append(output)

  if output_h is False:
    nodes_outputs.append('')
  else:
    h_shape = [num_directions, batch_size, hidden_size]
    y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, h_shape)
    nodes_outputs.append('Y_h')
    graph_outputs.append(y_h)

  if output_c is False:
    nodes_outputs.append('')
  else:
    c_shape = [num_directions, batch_size, hidden_size]
    y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT, c_shape)
    nodes_outputs.append('Y_c')
    graph_outputs.append(y_c)

  # lstm node
  node = onnx.helper.make_node(
      'LSTM',
      inputs=nodes_inputs,
      outputs=nodes_outputs,
      **attributes_dict
  )
  nodes.append(node)

  # graph
  graph_def = helper.make_graph(
      nodes,
      'test-model',
      graph_inputs,
      graph_outputs,
      initializer=initializers
  )

  model_def = helper.make_model(graph_def, producer_name='onnx')
  model_path = "test.onnx"
  onnx.save(model_def, model_path)

  sess = ort.InferenceSession(model_path)
  input_dict = {}
  input_dict['input'] = x
  outputs = sess.run(None, input_dict)

  return outputs


def run_native_v1(x: np.ndarray, w: np.ndarray, r: np.ndarray, direction, hidden_size, sequence_len, batch_size, embbeding_size, bias: np.ndarray, sequence_lens, initial_h: np.ndarray, initial_c: np.ndarray):
  num_directions = 2 if direction == 'bidirectional' else 1
  # input_shape = [sequence_len, batch_size, embbeding_size]
  # w_shape = [num_directions, 4 * hidden_size, embbeding_size]
  # r_shape = [num_directions, 4 * hidden_size, hidden_size]
  # bias_shape = [num_directions, 8 * hidden_size]
  # initial_c_shape = [num_directions, batch_size, hidden_size]
  # initial_h_shape = [num_directions, batch_size, hidden_size]
  output_y_shape = [sequence_len, num_directions, batch_size, hidden_size]
  h_shape = [num_directions, batch_size, hidden_size]
  c_shape = [num_directions, batch_size, hidden_size]
  output_h_tmp = np.copy(initial_h)
  output_c_tmp = np.copy(initial_c)
  output_y = np.zeros(output_y_shape).astype(np.float32)
  output_h = np.zeros(h_shape).astype(np.float32)
  output_c = np.zeros(c_shape).astype(np.float32)
  seq_len_loop = list()
  for l in range(sequence_len):
    seq_len_loop.append(l)
  if direction == 'reverse':
    seq_len_loop = seq_len_loop[::-1]
  for d in range(num_directions):
    if d == 1:
      seq_len_loop = seq_len_loop[::-1]
    for b in range(batch_size):
      for l in seq_len_loop:
        # xit = sigmoid(Xt*(Wi^T) + Wbi)
        # out_mul1: xt * w
        # out_mul2: h * r
        # [1,embbeding_size] @ [embbeding_size, 4 * hidden_size] = [4 * hidden_size]
        out_mul1 = x[l, b] @ w[d].T
        out_mul1 += bias[d, 0:4 * hidden_size]

        # [1,hidden_size] @ [hidden_size, 4 * hidden_size] = [4 * hidden_size]
        out_mul2 = output_h_tmp[d, b] @ r[d].T
        out_mul2 += bias[d, 4 * hidden_size:]

        out_mul1 += out_mul2

        # ft = sigmoid(g[2])
        out_mul1[hidden_size * 2:hidden_size * (2 + 1)] = sigmoid(out_mul1[hidden_size * 2:hidden_size * (2 + 1)])

        # ct = init_c * ft
        out_mul1[hidden_size * 2:hidden_size * (2 + 1)] = out_mul1[hidden_size * 2:hidden_size * (2 + 1)] * output_c_tmp[d, b]

        # it = sigmoid(g[0])
        out_mul1[hidden_size * 0:hidden_size * (0 + 1)] = sigmoid(out_mul1[hidden_size * 0:hidden_size * (0 + 1)])

        # c_t = tanh(g[3])
        out_mul1[hidden_size * 3:hidden_size * (3 + 1)] = tanh(out_mul1[hidden_size * 3:hidden_size * (3 + 1)])

        # c_t_it = it * c_t
        out_mul1[hidden_size * 0:hidden_size * (0 + 1)] = out_mul1[hidden_size * 0:hidden_size * (0 + 1)] * \
                                                          out_mul1[hidden_size * 3:hidden_size * (3 + 1)]

        # ct = ct + c_t_it
        output_c_tmp[d, b] = out_mul1[hidden_size * 2:hidden_size * (2 + 1)] + \
                             out_mul1[hidden_size * 0:hidden_size * (0 + 1)]

        # ot = sigmoid(g[1])
        out_mul1[hidden_size * 1:hidden_size * (1 + 1)] = sigmoid(out_mul1[hidden_size * 1:hidden_size * (1 + 1)])

        # tanh_ct = tanh(ct_o)
        out_mul1[hidden_size * 3:hidden_size * (3 + 1)] = tanh(output_c_tmp[d, b])

        # ht = tanh_ct * ot 
        output_h_tmp[d, b] = out_mul1[hidden_size * 3:hidden_size * (3 + 1)] *\
                             out_mul1[hidden_size * 1:hidden_size * (1 + 1)]

        output_y[l, d, b] = output_h_tmp[d, b]

        if l == seq_len_loop[-1]:
          output_h[d, b] = output_h_tmp[d, b]
          output_c[d, b] = output_c_tmp[d, b]

    return (output_y, output_h, output_c)


# def run_native_v2(x: np.ndarray, w: np.ndarray, r: np.ndarray, direction, hidden_size, sequence_len, batch_size, embbeding_size, bias: np.ndarray, sequence_lens, initial_h: np.ndarray, initial_c: np.ndarray):
#   num_directions = 2 if direction == 'bidirectional' else 1
#   # input_shape = [sequence_len, batch_size, embbeding_size]
#   # w_shape = [num_directions, 4 * hidden_size, embbeding_size]
#   # r_shape = [num_directions, 4 * hidden_size, hidden_size]
#   # bias_shape = [num_directions, 8 * hidden_size]  # bias 则是l r也合并在一起.
#   # initial_c_shape = [num_directions, batch_size, hidden_size]
#   # initial_h_shape = [num_directions, batch_size, hidden_size]
#   output_y_shape = [sequence_len, num_directions, batch_size, hidden_size]
#   h_shape = [num_directions, batch_size, hidden_size]
#   c_shape = [num_directions, batch_size, hidden_size]
#   output_h_tmp = np.copy(initial_h)
#   output_c_tmp = np.copy(initial_c)
#   output_y = np.zeros(output_y_shape).astype(np.float32)
#   output_h = np.zeros(h_shape).astype(np.float32)
#   output_c = np.zeros(c_shape).astype(np.float32)
#   seq_len_loop = list()
#   for l in range(sequence_len):
#     seq_len_loop.append(l)
#   if direction == 'reverse':
#     seq_len_loop = seq_len_loop[::-1]
#   for d in range(num_directions):
#     if d == 1:
#       seq_len_loop = seq_len_loop[::-1]
#     for b in range(batch_size):
#       for l in seq_len_loop:
#         """ 
#         it = f(Xt @ (Wi^T) + Ht-1 @ (Ri^T) + Wbi + Rbi)
#         ft = f(Xt @ (Wf^T) + Ht-1 @ (Rf^T) + Wbf + Rbf)
#         ct = g(Xt @ (Wc^T) + Ht-1 @ (Rc^T) + Wbc + Rbc)
#         ot = f(Xt @ (Wo^T) + Ht-1 @ (Ro^T) + Wbo + Rbo)
#         Ct = ft * Ct-1 + it * ct
#         Ht = ot * h(Ct)
#         """
#         # xit = sigmoid(Xt*(Wi^T) + Wbi)
#         # out_mul1: xt * w
#         # out_mul2: h * r
#         # [1,embbeding_size] @ [embbeding_size, 4 * hidden_size] = [4 * hidden_size]
#         # [1,hidden_size] @ [hidden_size, 4 * hidden_size] = [4 * hidden_size]
#         gates = (x[l, b] @ w[d].T) + bias[d, 0:4 * hidden_size] + \
#             (output_h_tmp[d, b] @ r[d].T) + bias[d, 4 * hidden_size:]
#         g0, g1, g2, g3 = np.split(gates, 4, 0)
#         it, ft, ct, ot = sigmoid(g0), sigmoid(g1), tanh(g2), sigmoid(g3)
#         output_c_tmp[d, b] = ft * output_h_tmp[d, b] + it * ct
#         output_h_tmp[d, b] = tanh(output_c_tmp[d, b]) * ot 

#         output_y[l, d, b] = output_h_tmp[d, b]

#         if l == seq_len_loop[-1]:
#           output_h[d, b] = output_h_tmp[d, b]
#           output_c[d, b] = output_c_tmp[d, b]

#     return (output_y, output_h, output_c)


param_directions = [
    'forward',
    # 'reverse',
    # 'bidirectional'
]

param_hidden_sizes = [
    3,
    16
]

param_sequence_lens = [
    2,
    8,
]

param_batch_sizes = [
    1,
]

param_embbeding_sizes = [
    1,
    16,
]

param_biases = [
    True,
    False,
]

param_sequence_lenses = [
    None,
]

param_initial_hs = [
    True,
    False,
]

param_initial_cs = [
    True,
    False,
]

param_Ys = [
    True,
]


param_Y_hs = [
    True,
    # False,
]

param_Y_cs = [
    True,
    # False,
]


@pytest.mark.parametrize('direction', param_directions)
@pytest.mark.parametrize('hidden_size', param_hidden_sizes)
@pytest.mark.parametrize('sequence_len', param_sequence_lens)
@pytest.mark.parametrize('batch_size', param_batch_sizes)
@pytest.mark.parametrize('embbeding_size', param_embbeding_sizes)  # input_size
@pytest.mark.parametrize('bias', param_biases)
@pytest.mark.parametrize('sequence_lens', param_sequence_lenses)
@pytest.mark.parametrize('initial_h', param_initial_hs)
@pytest.mark.parametrize('initial_c', param_initial_cs)
@pytest.mark.parametrize('Y', param_Ys)
@pytest.mark.parametrize('Y_h', param_Y_hs)
@pytest.mark.parametrize('Y_c', param_Y_cs)
def test_lstm(direction, hidden_size, sequence_len, batch_size, embbeding_size, bias: bool, sequence_lens, initial_h: bool, initial_c: bool, Y: bool, Y_h: bool, Y_c: bool, request):
  num_directions = 2 if direction == 'bidirectional' else 1
  input_shape = [sequence_len, batch_size, embbeding_size]
  w_shape = [num_directions, 4 * hidden_size, embbeding_size]
  r_shape = [num_directions, 4 * hidden_size, hidden_size]
  bias_shape = [num_directions, 8 * hidden_size]  # bias 则是l r也合并在一起.
  initial_c_shape = [num_directions, batch_size, hidden_size]
  initial_h_shape = [num_directions, batch_size, hidden_size]

  # 2. make tensor
  x = np.random.rand(*input_shape).astype(np.float32)
  w = np.random.rand(*w_shape).astype(np.float32) * 2 - 1
  r = np.random.rand(*r_shape).astype(np.float32) * 2 - 1

  b = np.zeros(bias_shape).astype(np.float32)
  if bias:
    b = np.random.rand(*bias_shape).astype(np.float32)

  init_h = np.zeros(initial_h_shape).astype(np.float32)
  if initial_h:
    init_h = np.random.rand(*initial_h_shape).astype(np.float32)

  init_c = np.zeros(initial_c_shape).astype(np.float32)
  if initial_c:
    init_c = np.random.rand(*initial_c_shape).astype(np.float32)

  # 3. run reference

  refernce_outputs = run_reference(x, w, r, direction, hidden_size, sequence_len,
                                   batch_size, embbeding_size, b, sequence_lens, init_h, init_c, Y, Y_h, Y_c)
  ref_output_y: np.ndarray = refernce_outputs[0]
  ref_output_h: np.ndarray = refernce_outputs[1]
  ref_output_c: np.ndarray = refernce_outputs[2]

  native_outputs = run_native_v1(x, w, r, direction, hidden_size, sequence_len,
                                 batch_size, embbeding_size, b, sequence_lens, init_h, init_c)
  actual_output_y: np.ndarray = native_outputs[0]
  actual_output_h: np.ndarray = native_outputs[1]
  actual_output_c: np.ndarray = native_outputs[2]
  assert (np.allclose(ref_output_y, actual_output_y, atol=1e-6))
  assert (np.allclose(ref_output_h, actual_output_h, atol=1e-6))
  assert (np.allclose(ref_output_c, actual_output_c, atol=1e-6))

  # native_outputs = run_native_v2(x, w, r, direction, hidden_size, sequence_len,
  #                             batch_size, embbeding_size, b, sequence_lens, init_h, init_c)
  # actual_output_y: np.ndarray = native_outputs[0]
  # actual_output_h: np.ndarray = native_outputs[1]
  # actual_output_c: np.ndarray = native_outputs[2]
  # assert (np.allclose(ref_output_y, actual_output_y, atol=1e-6))
  # assert (np.allclose(ref_output_h, actual_output_h, atol=1e-6))
  # assert (np.allclose(ref_output_c, actual_output_c, atol=1e-6))

if __name__ == "__main__":
  pytest.main(['-vv', __file__, '-s'])

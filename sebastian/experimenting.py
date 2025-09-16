import minitorch
import numpy as np

t35 = minitorch.TensorData([x for x in range(15)], (3, 5))
t15 = minitorch.TensorData([x for x in range(5)], (1, 5))
t55 = minitorch.TensorData([x for x in range(25)], (5, 5))
t551 = minitorch.TensorData([x for x in range(25)], (5, 5, 1))
t515 = minitorch.TensorData([x for x in range(25)], (5, 1, 5))
t234 = minitorch.TensorData([x for x in range(24)], (2, 3, 4))

for i in t515.indices():
    out_i = np.array([-1, -1, -1])
    minitorch.broadcast_index(np.array(i), t515._shape, t551._shape, out_i)
    # print(i, out_i)

permutation = minitorch.Tensor(minitorch.TensorData([1, 2, 0], (3, 1)), backend= minitorch.SimpleBackend)
t_t234 = minitorch.Tensor(t234, backend= minitorch.SimpleBackend)

# print(t_t234.shape, t_t234)

permuted = minitorch.Permute.apply(t_t234, permutation)

# print(permuted.shape, permuted)

def permute(a: minitorch.Tensor) -> minitorch.Tensor:
    return a.permute(2, 1, 0)

minitorch.grad_check(permute, t_t234)

# print(tensor_data.to_string())
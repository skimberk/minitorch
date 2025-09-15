import minitorch
import numpy as np

t35 = minitorch.TensorData([x for x in range(15)], (3, 5))
t15 = minitorch.TensorData([x for x in range(5)], (1, 5))
t55 = minitorch.TensorData([x for x in range(25)], (5, 5))
t551 = minitorch.TensorData([x for x in range(25)], (5, 5, 1))
t515 = minitorch.TensorData([x for x in range(25)], (5, 1, 5))

for i in t515.indices():
    out_i = np.array([-1, -1, -1])
    minitorch.broadcast_index(np.array(i), t515._shape, t551._shape, out_i)
    print(i, out_i)

# print(tensor_data.to_string())
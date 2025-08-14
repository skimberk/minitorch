import minitorch

data = [x for x in range(15)]
tensor_data = minitorch.TensorData(data, (3, 5), (5, 1))

print(tensor_data.to_string())
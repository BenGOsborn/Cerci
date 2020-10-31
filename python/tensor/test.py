from tensor_new import Tensor

x = Tensor([2, 2, 2, 2], [2, 2], track_grad=True)
y = Tensor([3, 3, 3, 3], [2, 2], track_grad=True)

z = x * y * y
z.backwards()

print(y.grad)
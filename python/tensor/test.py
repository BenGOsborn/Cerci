from tensor_expressions import Tensor

x = Tensor([2, 2, 2, 2], [2, 2])
y = Tensor([3, 3, 3, 3], [2, 2])

z = x * y + x
z.backwards()
print(x.grad)

# This is why the tensor needs to have all of these options built into it recursively
from tensor_new import Tensor

x = Tensor([2, 2, 2, 2], [2, 2], track_grad=True)
y = Tensor([3, 3, 3, 3], [2, 2], track_grad=True)

z = x * y
z.backwards()

# This is why the tensor needs to have all of these options built into it recursively

# New tensor architecture:
#   Contain all of the tensor features and functions
#   Track the gradients and the operations that happen and when added create a new tensor and store the two previous ones in it

# Have a chain option meaning if it is a chain link then we do not calculate the gradient but if it is not a chain then we do calculate the gradient system?
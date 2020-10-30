class AddElementwise:

    @staticmethod
    def forward(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        new_tensor = [a+b for a, b in zip(matrix_a.tensor, matrix_b.tensor)]

        return Tensor(new_tensor, matrix_a.shape, left=matrix_a, right=matrix_b, track_grad=(matrix_a.track_grad or matrix_b.track_grad), operator=AddElementwise) # Use matrix_a.requires_grad or matrix_b.requires_grad
        
    @staticmethod
    def dda(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        ret_tensor = [1 for _ in range(matrix_a.size)]

        # What operator should it return here and its not like it needs to track the gradient it just needs to be able to track itself
        return Tensor(ret_tensor, matrix_a.shape, left=None, right=None, track_grad=False, operator=None)

    @staticmethod
    def ddb(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        ret_tensor = [1 for _ in range(matrix_a.size)]

        return Tensor(ret_tensor, matrix_a.shape, left=None, right=None, track_grad=False, operator=None)

class MultiplyElementwise:

    @staticmethod
    def forward(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        new_tensor = [a*b for a, b in zip(matrix_a.tensor, matrix_b.tensor)]

        return Tensor(new_tensor, matrix_a.shape, left=matrix_a, right=matrix_b, track_grad=(matrix_a.track_grad or matrix_b.track_grad), operator=MultiplyElementwise)

    # But with these does it need to track the left and the right or no?

    @staticmethod
    def dda(matrix_a, matrix_b):
        return Tensor(matrix_a.tensor.copy(), matrix_a.shape.copy(), left=None, right=None, track_grad=False, operator=None)

    @staticmethod
    def ddb(matrix_a, matrix_b):
        return Tensor(matrix_b.tensor.copy(), matrix_b.shape.copy(), left=None, right=None, track_grad=False, operator=None)

# This is going to contain all of the information about the tensor relating to the gradient
class TensorGrad:
    def __init__(self, left, right, track_grad, operator):
        self.track_grad = False
        self.operator = None
        self.grad = None

        self.left = left
        self.right = right

class Tensor(TensorGrad):
    def __init__(self, tensor, shape, left=None, right=None, track_grad=False, operator=None): # The left and the right will contain the links to the other nodes in the tree
        self.dims = len(shape)
        self.size = len(tensor)

        check_length = 1
        for i in range(self.dims):
            check_length *= shape[i]
        assert(check_length == self.size)

        self.tensor = tensor
        self.shape = shape

        super().__init__(left, right, track_grad, operator)
    
    def __str__(self):
        return self.__string()

    # There is probably a more efficient way to do this
    def __string(self, index=-1, position=0):
        if (abs(index) == self.dims):
            mat = "[ "
            for i in range(self.shape[0]):
                mat += f"{self.tensor[position + i]} "
            mat += "]"
            return mat
        
        mat_final = "[ "

        product = 1
        for i in range(self.dims + index):
            product *= self.shape[i]

        for i in range(self.shape[index]):
            mat_final += f"\n{abs(index) * ' '}{ self.__string(index-1, position+product*i)} "
        
        return f"{mat_final}\n{(abs(index) - 1) * ' '}]" if (index != -1) else f"{mat_final}\n]"

    def __add__(self, other):
        return AddElementwise.forward(self, other)

    def __mul__(self, other):
        return MultiplyElementwise.forward(self, other)

    def reset(self):
        self.grad = None

    # Now I need to do the actual backwards function
    def backwards(self, factors):
        pass
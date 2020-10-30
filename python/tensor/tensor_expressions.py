import tensor_operators
import tensor_base

class ExpressionBase:
    def __init__(self, op):
        self.op = op
        self.forwarded = None

    def __add__(self, other):
        return Expression(self, other, tensor_operators.AddElementwise)

    # Elementwise multiplication, use dot for a matrix multiplication
    def __mul__(self, other):
        return Expression(self, other, tensor_operators.MultiplyElementwise)

class Expression(ExpressionBase):
    def __init__(self, a, b, operator):
        super().__init__(operator)
        self.a = a
        self.b = b

    def reset(self):
        self.forwarded = None
        self.a.reset()
        self.b.reset()

    def forward(self):
        if (self.forwarded != None):
            return self.forwarded
        a = self.a.forward()
        b = self.b.forward()
        self.forwarded = self.op.forward(a, b)
        return self.forwarded

    def backwards(self, factors=1):
        if (self.a.requires_grad):
            self.a.backwards(factors=factors*self.op.dda(self.a.forward(), self.b.forward()))
        if (self.b.requires_grad):
            self.b.backwards(factors=factors*self.op.ddb(self.a.forward(), self.b.forward()))

class ExpressionSingle(ExpressionBase):
    def __init__(self, a, operator):
        super().__init__(operator)
        self.a = a

    def reset(self):
        self.forwarded = None
        self.a.reset()

    def forward(self):
        if (self.forwarded != None):
            return self.forwarded
        a = self.a.forward()
        self.forwarded = self.op.forward(a)
        return self.forwarded

    def backwards(self, factors=1):
        if (self.a.requires_grad):
            self.a.backwards(factors=factors*self.op.dda(self.a.forward()))

# But we want this tensor to be the foundation of all of the others
class Tensor(ExpressionBase):
    def __init__(self, tensor, shape, requires_grad=True):
        self.dims = len(shape)
        self.size = len(tensor)

        check_length = 1
        for i in range(self.dims):
            check_length *= shape[i]
        assert(check_length == self.size)

        self.tensor = tensor
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad = None
    
    def __str__(self):
        return self.__string()

    def __string(self, index=-1, position=0):
        # This is the base case that will occur if it is the last node in the tree
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

    def reshape(self, new_shape):
        assert(self.dims == len(new_shape))

        new_size = 1
        for i in range(len(new_shape)):
            new_size *= new_shape[i]
        assert(new_size == self.size)

        return Tensor(self.tensor.copy(), self.shape.copy())

    def reset(self):
        self.grad = None

    def forward(self):
        return self

    def backwards(self, factors):
        if (self.grad != None):
            self.grad = tensor_operators.AddElementwise.forward(factors, self)
            return
        self.grad = factors

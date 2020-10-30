import tensor_operators

# How can I add a requires grad into this base class?
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

    def backwards(self, factors=None):
        if (factors == None):
            factors = Tensor([1 for _ in range(self.forward().size)], self.forward().shape)
        self.a.backwards(factors=factors*self.op.dda(self.a.forward(), self.b.forward()))
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

    def backwards(self, factors=None):
        if (factors == None):
            factors = Tensor([1 for _ in range(self.forward().size)], self.forward().shape)
        self.a.backwards(factors=factors*self.op.dda(self.a.forward()))

class Tensor(ExpressionBase):
    def __init__(self, tensor, shape):
        self.dims = len(shape)
        self.size = len(tensor)

        check_length = 1
        for i in range(self.dims):
            check_length *= shape[i]
        assert(check_length == self.size)

        self.tensor = tensor
        self.shape = shape
        self.grad = None
    
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

    def reset(self):
        self.grad = None

    def forward(self):
        return self

    # But the factos being parsed through should be a tensor not an expression object...
    def backwards(self, factors):
        if (self.grad != None):
            self.grad = tensor_operators.AddElementwise.forward(self, factors)
            return
        self.grad = factors.forward()

import tensor_operators
import tensor

class ExpressionBase:
    def __init__(self, op):
        self.op = op
        self.forwarded = None

    def __add__(self, other):
        return Expression(self, other, tensor_operators.AddElementwise)

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
        self.a.backwards(factors=factors*self.op.dda(self.a.forward(), self.b.forward()))
        self.b.backwards(factors=factors*self.op.ddb(self.a.forward(), self.b.forward()))

tensor1 = tensor.Tensor([1, 2, 3, 4], [2, 2])
tensor2 = tensor.Tensor([1, 2, 3, 4], [2, 2])
print(tensor_operators.AddElementwise().forward(tensor1, tensor2))
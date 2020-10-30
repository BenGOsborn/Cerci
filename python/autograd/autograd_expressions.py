from autograd_operators import *

class ExpressionBase:
    def __init__(self, op):
        self.op = op
        self.forwarded = None

    def __add__(self, other):
        return ExpressionDouble(self, other, OperatorAdd)

    def __sub__(self, other):
        return ExpressionDouble(self, other, OperatorSubtract)

    def __mul__(self, other):
        return ExpressionDouble(self, other, OperatorMultiply)

    def __truediv__(self, other):
        return ExpressionDouble(self, other, OperatorDivide)

    def __pow__(self, other):
        return ExpressionDouble(self, other, OperatorPower)

class ExpressionSingle(ExpressionBase):
    def __init__(self, a, op):
        super().__init__(op)
        self.a = a

    # Make it so it doesnt need to have forward called and can just be called by int() or by str()
    def forward(self):
        if (self.forwarded != None):
            return self.forwarded
        a = self.a.forward()
        self.forwarded = self.op.eval(a)
        return self.forwarded

    def backwards(self, factors=1):
        self.a.backwards(factors=factors*self.op.dda())

    def reset(self):
        self.a.reset()

class ExpressionDouble(ExpressionBase):
    def __init__(self, a, b, op):
        super().__init__(op)
        self.a = a
        self.b = b

    # Instead have it so calling forward instead is just the same as printing the string or calling the string as an int
    def forward(self):
        if (self.forwarded != None):
            return self.forwarded
        a = self.a.forward()
        b = self.b.forward()
        self.forwarded = self.op.eval(a, b)
        return self.forwarded

    def backwards(self, factors=1):
        self.a.backwards(factors=factors*self.op.dda(self.a.forward(), self.b.forward()))
        self.b.backwards(factors=factors*self.op.ddb(self.a.forward(), self.b.forward()))

    def reset(self):
        self.a.reset()
        self.b.reset()

class Variable(ExpressionBase):
    def __init__(self, value):
        self.value = value
        self.grad = None

    def reset(self):
        self.grad = None

    def forward(self):
        return self.value

    def backwards(self, factors):
        if (self.grad != None):
            self.grad += factors
            return
        self.grad = factors

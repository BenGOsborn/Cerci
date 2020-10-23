import math

class Operator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class OperatorAdd(Operator):
    def __init__(self, a, b):
        super().__init__(a, b)

    def eval(self):
        return self.a.forward() + self.b.forward()

    def dda(self):
        return 1

    def ddb(self):
        return 1

class OperatorSubtract(Operator):
    def __init__(self, a, b):
        super().__init__(a, b)

    def eval(self):
        return self.a.forward() - self.b.forward()

    def dda(self):
        return 1

    def ddb(self):
        return -1

class OperatorMultiply(Operator):
    def __init__(self, a, b):
        super().__init__(a, b)

    def eval(self):
        return self.a.forward() * self.b.forward()

    def dda(self):
        return self.b.forward()

    def ddb(self):
        return self.a.forward()

class OperatorDivide(Operator):
    def __init__(self, a, b):
        super().__init__(a, b)

    def eval(self):
        return self.a.forward() / self.b.forward()

    def dda(self):
        return 1 / self.b.forward()

    def ddb(self):
        return -self.a.forward() / (self.b.forward() * self.b.forward())

class OperatorPower(Operator):
    def __init__(self, a, b):
        super().__init__(a, b)

    def eval(self):
        return self.a.forward() ** self.b.forward()

    def dda(self):
        return self.b.forward() * (self.a.forward() ** (self.b.forward() - 1))

    def ddb(self):
        return math.log(self.a.forward()) * (self.a.forward() ** self.b.forward())

class Expression:
    def __init__(self, a, b, op):
        self.a = a
        self.b = b
        self.op = op(a, b)
        self.forwarded = None

    def __add__(self, other):
        return Expression(self, other, OperatorAdd)

    def __sub__(self, other):
        return Expression(self, other, OperatorSubtract)

    def __mul__(self, other):
        return Expression(self, other, OperatorMultiply)

    def __truediv__(self, other):
        return Expression(self, other, OperatorDivide)

    def __pow__(self, other):
        return Expression(self, other, OperatorPower)

    def backwards(self, factors=1):
        self.a.backwards(factors=factors*self.op.dda())
        self.b.backwards(factors=factors*self.op.ddb())

    def reset(self):
        self.a.reset()
        self.b.reset()

    def forward(self):
        if (self.forwarded != None):
            return self.forwarded

        self.forwarded = self.op.eval()
        return self.forwarded

class Variable(Expression):
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

x = Variable(0)
sigmoid = Variable(1) / ( Variable(1) + ( Variable(math.e) ** ( Variable(-1) * x ) ) ) 

sigmoid.backwards()

print(x.grad)
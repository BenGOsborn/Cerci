class Add:
    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.forwarded = None

        self.grad = None
        self.grad_a = None
        self.grad_b = None

    def forward(self):
        self.forwarded = self.a.forward() + self.b.forward()
        return self.forwarded

    def gradA(self):
        self.grad_a = 1
        self.a.grad_a = self.grad_a * self.a.gradA()
        self.a.grad_b = self.grad_a * self.a.gradB()
        return self.grad_a

    def gradB(self):
        self.grad_b = 1
        self.b.grad_a = self.grad_b * self.b.gradA()
        self.b.grad_b = self.grad_b * self.b.gradB()
        return self.grad_b

    def backwards(self):
        self.gradA()
        self.gradB()

class Var:
    def __init__(self, val):
        self.forwarded = val
        
        self.grad = None
        self.grad_a = None
        self.grad_b = None

    def forward(self):
        return self.forwarded

    def gradA(self):
        self.grad_a = 1
        return self.grad_a

    def gradB(self):
        self.grad_b = 0
        return self.grad_b

a = Var(3)
b = Var(2)
c = Var(8)
d = Var(5)

e = Add(a, b)
f = Add(c, d)

g = Add(e, f)

g.forward()
g.backwards()

# It appears that the gradients of what it should be has been shifted down, so the deriv of g with respect to a is a little bit broken due to finding grad a and grad b and such
# This can be fixed by replacing seperate gradients to get the gradient with respect to that val
# To fix this as well start with the root derivative of each to be 1 and then mulltiply it out respectively
print(e.grad_b)
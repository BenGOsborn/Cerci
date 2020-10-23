from autograd_expressions import Variable
from autograd_functions import *

a = Variable(5)
b = Variable(2)

c = a + b ** b

c.backwards()

print(b.grad)
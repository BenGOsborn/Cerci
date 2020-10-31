from autograd_expressions import Variable
from autograd_functions import *

a = Variable(2)
b = Variable(3)

c = a * b

c.backwards()

print(a.grad)
from math import exp, tanh
from matrix import Matrix

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+exp(-x))

def relu(x, deriv=False):
    if deriv:
        return 1 if x > 0 else 0.1
    return max(0.1*x, x)
    
# This is the ADAM learning rate paramater which will change depending on the error matrix it receives
def adam(errors):
    return 0.5*abs(tanh(4*errors.average()))
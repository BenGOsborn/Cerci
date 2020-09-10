from math import exp, tanh
from matrix import Matrix, subtract

# Activation functions
def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+exp(-x))

def relu(x, deriv=False):
    if deriv:
        return 1 if x > 0 else 0.1
    return max(0.1*x, x)

def softmax(val, vals=None, deriv=False):
    if deriv:
        return val*(1-val)
    return exp(val)/sum([exp(x) for x in vals])
    
# Loss functions

# Perhaps we should be writing these in terms of the matrices for simplicity
def meanSquared(predicted_matrix, actual_matrix):
    return subtract(actual_matrix, predicted_matrix)

def crossEntropy(predicted_matrix, actual_matrix):
    return

# This is the ADAM learning rate paramater which will change depending on the error matrix it receives
def adam(errors):
    return 0.5*abs(tanh(4*errors.average()))
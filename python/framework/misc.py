from math import exp, tanh, log
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
def meanSquared(predicted_matrix, actual_matrix):
    return subtract(actual_matrix, predicted_matrix)

def crossEntropy(predicted_matrix, actual_matrix):
    fn = lambda predicted, actual: -1*(actual/predicted) + (1-actual)/(1-predicted)

    flat_pred = Matrix(arr=predicted_matrix)
    flat_pred.flatten()
    flat_act = Matrix(arr=actual_matrix)
    flat_act.flatten()

    matrix_new = [fn(pred, act) for pred, act in zip(flat_pred.returnMatrix()[0], flat_act.returnMatrix()[0])]
    ret_matrix = Matrix(matrix_new)
    ret_matrix.transpose()

    return ret_matrix

# This is the ADAM learning rate paramater which will change depending on the error matrix it receives
def adam(errors):
    return 0.5*abs(tanh(4*errors.average()))
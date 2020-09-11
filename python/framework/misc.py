from math import exp, tanh, log
from matrix import Matrix, subtract

# Activation functions
def sigmoid(x, vals=None, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+exp(-x))

def relu(x, vals=None, deriv=False):
    if deriv:
        return 1 if x > 0 else 0.1
    return max(0.1*x, x)

def softmax(val, vals=None, deriv=False):
    if deriv:
        return val*(1-val)
    vals.flatten()
    vals = vals.returnMatrix()[0]
    return exp(val)/sum([exp(x) for x in vals])
    
# Loss functions
def meanSquared(predicted, actual):
    return predicted-actual

def crossEntropy(predicted, actual):
    return -1*(actual/predicted) + (1-actual)/(1-predicted)

# This is the ADAM learning rate paramater which will change depending on the error matrix it receives
def adam(errors):
    return 0.5*abs(tanh(4*errors.average()))

# Returns the back errors
def backErrors(loss, activation, predicted, training, shape):
    # Going to return the errors*sigmoids
    predObj = Matrix(arr=predicted.returnMatrix())
    trainObj = Matrix(arr=training.returnMatrix())

    predObj.flatten()
    trainObj.flatten()

    predMat = predObj.returnMatrix()[0]
    trainMat = trainObj.returnMatrix()[0]

    # The loss function does not work here because its to work only for matrices
    mat_partial = [loss(pred, act)*activation(pred, deriv=True) for pred, act in zip(predMat, trainMat)]

    newMat = Matrix(arr=mat_partial)

    # Returns the data in the shaped data
    newMat.reshape(shape[0], shape[1])

    return newMat


from math import exp, tanh, log
from matrix import Matrix

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

# Optimizers
def applyMomentum(p_prev, beta1, gradient):
    p = p_prev*beta1 + (1-beta1)*gradient
    return p

# This value should actually NEVER be negative what is going on then?
def applyRMS(rms_prev, beta2, gradient):
    rms = rms_prev*beta2 + (1-beta2)*(gradient**2)
    return rms

def applyCorrection(param, beta, iteration):
    corrected = param/(1-beta**iteration)
    return corrected

# Maybe its not applying it correctly in the feedForward

# If I add other optimizers Im going to have to map the optimizers to have the same imput parameters with 'n=b' notation

def adam(pPrev, rmsPrev, gradients, beta1, beta2, epsilon, iteration):
    # Define functions up here

    gradRaw = gradients.returnMatrix()
    pPrevRaw = pPrev.returnMatrix()
    rmsPrevRaw = rmsPrev.returnMatrix()
    gradSize = gradients.size()

    # Now with this we can do a calculation for the new rms and the new momentums and then return those aswell as a matrix
    pRaw = [] # Return this as matrix
    rmsRaw = [] # Return this as matrix
    adamRaw = []
    for y in range(gradSize[0]):
        tempArrP = []
        tempArrRMS = []
        tempArrAdam = []
        for x in range(gradSize[1]):
            momentum = applyMomentum(pPrevRaw[y][x], beta1, gradRaw[y][x])
            rms = applyRMS(rmsPrevRaw[y][x], beta2, gradRaw[y][x]) # Apply RMS here

            tempArrP.append(momentum)
            tempArrRMS.append(rms)

            momentumCorrected = applyCorrection(momentum, beta1, iteration)
            rmsCorrected = applyCorrection(rms, beta2, iteration)
            # The RMS should actually never be negative
            adam = momentumCorrected / (rmsCorrected**(0.5) + epsilon) # This must have generated a complex number, why?
            tempArrAdam.append(adam)

        pRaw.append(tempArrP)
        rmsRaw.append(tempArrRMS)
        adamRaw.append(tempArrAdam)

    p = Matrix(arr=pRaw)
    rms = Matrix(arr=rmsRaw)
    adam = Matrix(arr=adamRaw)

    # Returns the momentum, the rms and the adam
    return p, rms, adam
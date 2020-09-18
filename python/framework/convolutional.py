import matrix
from misc import applyActivationGradient
from math import ceil

def kernel(inMatrix, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols):
    size = inMatrix.size()
    
    retMatrix = []
    for rowNum in range(size[0]-kernel_size_rows+1):
        for colNum in range(size[1]-kernel_size_cols+1):
            if ((rowNum % step_size_rows == 0) and (colNum % step_size_cols == 0)):
                tempTrix = inMatrix.cut(rowNum, rowNum+kernel_size_rows, colNum, colNum+kernel_size_cols).flatten().returnMatrix()[0]
                retMatrix.append(tempTrix)

    return matrix.Matrix(arr=retMatrix)

def weightedKernel(inMatrix, kernelMatrix, step_size_rows, step_size_cols):
    kernel_size_rows = kernelMatrix.size()[0]
    kernel_size_cols = kernelMatrix.size()[1]

    kernelled = kernel(inMatrix, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols) 
    weightedKernel = matrix.multiplyMatrices(kernelled, kernelMatrix.flatten().transpose())

    new_sizeRows = ceil((inputs.size()[0] - kernel_size_rows + 1) / step_size_rows)
    new_sizeCols = ceil((inputs.size()[1] - kernel_size_cols + 1) / step_size_cols)

    return weightedKernel.reshape(new_sizeRows, new_sizeCols)
    # Where else has there been this problem with the reshape where it was flattening the object?

class Convolutional:
    def __init__(self, weight_set, bias, step_size_rows, step_size_cols, activation_func):
        self.weights = weight_set
        self.bias = bias
        self.activation_func = activation_func

        self.kernel_size_rows = weight_set.size()[0]
        self.kernel_size_cols = weight_set.size()[1]
        self.step_size_rows = step_size_rows
        self.step_size_cols = step_size_cols

        self.pWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.iteration = 0

    def reinit(self):
        self.pWeight = matrix.Matrix(dims=self.pWeight.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=self.rmsWeight.size(), init=lambda: 0)
        self.iteration = 0

    def predict(self, inputs):
        # Test how the zero creation works

        wKernal = weightedKernel(inputs, self.weights, self.step_size_rows, self.step_size_cols)
        biasArray = matrix.Matrix(dims=[wKernal.size()[0], wKernal.size()[1]], init=lambda: self.bias.returnMatrix()[0][0])

        out = matrix.add(wKernal, biasArray)

        outCpy = out.clone() # This is required for the softmax function
        out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        return out

    def train(self, input_set_raw, predicted, errors_raw, optimizer, learn_rate=0.5):
        self.iteration += 1

        errors = applyActivationGradient(self.activation_func, errors_raw, predicted)


        # I need some way of mapping the error from the previous to the new error while filling in the gaps
        # I might have to train the network using the larger error values to be fed forward so that the size of the values between layers works

    def returnNetwork(self):
        pass

from misc import relu
weights = matrix.Matrix(dims=[3, 3], init=lambda: 0.5)
bias = matrix.Matrix(arr=[1])
inputs = matrix.Matrix(dims=[5, 5], init=lambda: 2)

x = Convolutional(weights, bias, 2, 2, relu)

prediction = x.predict(inputs)
prediction.print()
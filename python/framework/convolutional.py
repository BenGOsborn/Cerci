import matrix
from misc import backErrors
from math import ceil

class Convolutional:
    def __init__(self, weight_set, step_size_rows, step_size_cols, activation_func):
        self.weights = weight_set
        self.activation_func = activation_func

        self.step_size_rows = step_size_rows
        self.step_size_cols = step_size_cols

        self.pWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.iteration = 0

    def reinit(self):
        self.pWeight = matrix.Matrix(dims=self.pWeight.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=self.rmsWeight.size(), init=lambda: 0)
        self.iteration = 0

    def kernel(self, inMatrix):

        # Do I need to have seperate kernelling size for when we kernel things that are not the weights such as the errors

        # The kernel size is actually gonna be the size of the weights
        size = inMatrix.size()
        kernelSize = self.weights.size()

        retMatrix = []
        for rowNum in range(size[0]-kernelSize[0]+1):
            for colNum in range(size[1]-kernelSize[1]+1):
                if ((rowNum % self.step_size_rows == 0) and (colNum % self.step_size_cols == 0)):
                    tempTrix = inMatrix.cut(rowNum, rowNum+kernelSize[0], colNum, colNum+kernelSize[1]).flatten().returnMatrix()[0]
                    retMatrix.append(tempTrix)

        return matrix.Matrix(arr=retMatrix)

    def predict(self, inputs):
        # High level layer needs rewrite

        # Consider calling this function seperately for ease of use
        # Could also consider putting the step_size in the init part
        kernels = self.kernel(inputs) 

        out = matrix.multiplyMatrices(kernels, self.weights.flatten().transpose())

        outCpy = out.clone()
        out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        new_sizeRows = ceil((inputs.size()[0] - self.weights.size()[0] + 1) / self.step_size_rows)
        new_sizeCols = ceil((inputs.size()[1] - self.weights.size()[1] + 1) / self.step_size_cols)

        out = out.flatten().reshape(new_sizeRows, new_sizeCols)

        return out

    def train(self, input_set_raw, predicted, errors, optimizer, learn_rate):
        self.iteration += 1

        # Thisapplies the multiplication of the derivative of the activation function
        # Will this need to be transposed?
        errors = backErrors(self.activation_func, errors, predicted)

        # The input layer will just be the same kernelled values from the other layer
        kernel = self.kernel(input_set_raw)

        # Now here we do the error creation for the actual errors
        # Note that the errors parsed in should be a flattened layer
        kernelTranspose = kernel.transpose()
        w_AdjustmentsRaw = matrix.multiplyMatrices(kernelTranspose, errors)

        self.pWeight, self.rmsWeight, w_Adjustments = optimizer(self.pWeight, self.rmsWeight, w_AdjustmentsRaw, self.iteration)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate)
        self.weights = matrix.subtract(self.weights, w_Adjustments)

        # Now to get the errors for the previous layer I need to pad the error layer reshaped and then the 

    def returnNetwork(self):
        pass

from misc import relu
weights = matrix.Matrix(dims=[2, 2], init=lambda: 4)
inputs = matrix.Matrix(dims=[10, 10], init=lambda: 2)
x = Convolutional(weights, 1, 1, relu)
x.predict(inputs).print()
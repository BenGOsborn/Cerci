import matrix
import misc
from math import ceil

class Convolutional:
    def __init__(self, weight_set, activation_func):
        self.weights = weight_set
        self.activation_func = activation_func

        self.pWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.iteration = 0

    def reinit(self):
        self.pWeight = matrix.Matrix(dims=self.pWeight.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=self.rmsWeight.size(), init=lambda: 0)
        self.iteration = 0

    def kernel(self, inMatrix, step_size_rows, step_size_cols):
        # The kernel size is actually gonna be the size of the weights
        size = inMatrix.size()
        kernelSize = self.weights.size()

        retMatrix = []
        for rowNum in range(size[0]-kernelSize[0]+1):
            for colNum in range(size[1]-kernelSize[1]+1):
                if ((rowNum % step_size_rows == 0) and (colNum % step_size_cols == 0)):
                    tempTrix = inMatrix.cut(rowNum, rowNum+kernelSize[0], colNum, colNum+kernelSize[1]).flatten().returnMatrix()[0]
                    retMatrix.append(tempTrix)

        return matrix.Matrix(arr=retMatrix)

    def feedForward(self, inputs, step_size_rows, step_size_cols):
        # High level layer needs rewrite
        kernels = self.kernel(inputs, step_size_rows, step_size_cols)
        out = matrix.multiplyMatrices(kernels, self.weights.flatten().transpose())

        outCpy = out.clone()
        out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        new_sizeRows = ceil((inputs.size()[0] - self.weights.size()[0] + 1) / step_size_rows)
        new_sizeCols = ceil((inputs.size()[1] - self.weights.size()[1] + 1) / step_size_cols)

        out = out.flatten().reshape(new_sizeRows, new_sizeCols)

        return out

    def train(self):
        pass

    def returnNetwork(self):
        pass

weights = matrix.Matrix(dims=[7, 7], init=lambda: 4)
inputs = matrix.Matrix(dims=[10, 10], init=lambda: 2)
x = Convolutional(weights, misc.relu)
x.feedForward(inputs, 12, 12).print()
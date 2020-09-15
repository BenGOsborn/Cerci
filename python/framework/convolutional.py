import matrix
import misc

class Convolutional:
    def __init__(self, weight_set, bias_set, activation_func):
        self.weights = weight_set
        self.bias = bias_set
        self.activation_func = activation_func

        self.pWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.pBias = matrix.Matrix(dims=bias_set.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.rmsBias = matrix.Matrix(dims=bias_set.size(), init=lambda: 0)
        self.iteration = 0

    def reinit(self):
        self.pWeight = matrix.Matrix(dims=self.pWeight.size(), init=lambda: 0)
        self.pBias = matrix.Matrix(dims=self.pBias.size(), init=lambda: 0)
        self.rmsWeight = matrix.Matrix(dims=self.rmsWeight.size(), init=lambda: 0)
        self.rmsBias = matrix.Matrix(dims=self.rmsBias.size(), init=lambda: 0)
        self.iteration = 0

    def kernel(self, inMatrix, kernel_rows, kernel_cols, step_size_rows, step_size_cols):
        size = inMatrix.size()
        retMatrix = []
        for rowNum in range(size[0]-kernel_rows+1):
            for colNum in range(size[1]-kernel_cols+1):
                if ((rowNum % step_size_rows == 0) and (colNum % step_size_cols == 0)):
                    tempTrix = inMatrix.cut(rowNum, rowNum+kernel_rows, colNum, colNum+kernel_cols).flatten().returnMatrix()[0]
                    retMatrix.append(tempTrix)

        return matrix.Matrix(arr=retMatrix)

    def predict(self):
        pass

    def train(self):
        pass

    def returnNetwork(self):
        pass

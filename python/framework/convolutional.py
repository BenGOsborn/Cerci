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

    new_sizeRows = ceil((inMatrix.size()[0] - kernel_size_rows + 1) / step_size_rows)
    new_sizeCols = ceil((inMatrix.size()[1] - kernel_size_cols + 1) / step_size_cols)

    return weightedKernel.reshape(new_sizeRows, new_sizeCols)

def dilate(toDilate, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols):
    valArray = toDilate.returnMatrix()

    midGridSizeColLen = (len(valArray[0])-1)*(step_size_cols-1)+len(valArray[0])
    zeroArray = [0 for _ in range(midGridSizeColLen)]

    scaledHorizontal = []
    for y in range(len(valArray)):
        tempArr = []
        for x in range(len(valArray[0])):
            if (x != 0):
                for _ in range(step_size_rows-1):
                    tempArr.append(0)
            tempArr.append(valArray[y][x])
        scaledHorizontal.append(tempArr)

    scaledVertical = []
    for y in range(len(valArray)):
        if (y != 0):
            for _ in range(step_size_cols-1):
                scaledVertical.append(zeroArray)
        scaledVertical.append(scaledHorizontal[y])

    unpadded = matrix.Matrix(arr=scaledVertical)
    padded = unpadded.pad(pad_up=kernel_size_rows-1, pad_down=kernel_size_rows-1, pad_left=kernel_size_cols-1, pad_right=kernel_size_cols-1)

    return padded

class Convolutional:
    def __init__(self, weight_set, bias, step_size_rows, step_size_cols, activation_func):
        self.weights = weight_set
        self.bias = bias
        self.activation_func = activation_func

        self.kernel_size_rows = weight_set.size()[0]
        self.kernel_size_cols = weight_set.size()[1]
        self.step_size_rows = step_size_rows
        self.step_size_cols = step_size_cols

        self.pWeights = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.rmsWeights = matrix.Matrix(dims=weight_set.size(), init=lambda: 0)
        self.pBias = 0
        self.rmsBias = 0
        self.iteration = 0

    def predict(self, inputs):
        wKernel = weightedKernel(inputs, self.weights, self.step_size_rows, self.step_size_cols)
        biasSet = matrix.Matrix(dims=wKernel.size(), init=lambda: self.bias)

        out = matrix.add(wKernel, biasSet)

        outCpy = out.clone()
        out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        return out

    def train(self, input_set, predicted, errors_raw, optimizer, learn_rate=0.1):
        self.iteration += 1

        errors = applyActivationGradient(self.activation_func, errors_raw, predicted).transpose()

        kerneledTransposed = kernel(input_set, self.kernel_size_rows, self.kernel_size_cols, self.step_size_rows, self.step_size_rows).transpose()

        w_AdjustmentsRaw = matrix.multiplyMatrices(kerneledTransposed, errors).reshape(self.kernel_size_rows, self.kernel_size_cols)
        self.pWeights, self.rmsWeights, w_Adjustments = optimizer(self.pWeights, self.rmsWeights, w_AdjustmentsRaw, self.iteration)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate).reshape(self.kernel_size_rows, self.kernel_size_cols)
        self.weights = matrix.subtract(self.weights, w_Adjustments)
        
        errGradients = matrix.matrixSum(errors)
        self.pBias, self.rmsBias, b_Adjustments = optimizer(self.pBias, self.rmsBias, errGradients, self.iteration)
        self.bias = self.bias - learn_rate*b_Adjustments

        weightsFlipped = self.weights.rotate()
        errorsDilated = dilate(errors, self.kernel_size_rows, self.kernel_size_cols, self.step_size_rows, self.step_size_cols)
        h_Error = weightedKernel(errorsDilated, weightsFlipped, 1, 1)

        return h_Error

    def returnNetwork(self):
        return self.weights, self.pWeights, self.rmsWeights, self.bias, self.pBias, self.rmsBias
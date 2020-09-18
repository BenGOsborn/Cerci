import matrix
from misc import applyActivationGradient
from math import ceil

def kernel(inMatrix, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols, fillGaps=False):
    size = inMatrix.size()

    zeroRow = [0 for _ in range(kernel_size_rows*kernel_size_cols)]
    
    retMatrix = []
    for rowNum in range(size[0]-kernel_size_rows+1):
        for colNum in range(size[1]-kernel_size_cols+1):
            if ((rowNum % step_size_rows == 0) and (colNum % step_size_cols == 0)):
                tempTrix = inMatrix.cut(rowNum, rowNum+kernel_size_rows, colNum, colNum+kernel_size_cols).flatten().returnMatrix()[0]
                retMatrix.append(tempTrix)
            elif (fillGaps):
                retMatrix.append(zeroRow)

    return matrix.Matrix(arr=retMatrix)

def weightedKernel(inMatrix, kernelMatrix, step_size_rows, step_size_cols, fillGaps=False):
    kernel_size_rows = kernelMatrix.size()[0]
    kernel_size_cols = kernelMatrix.size()[1]

    kernelled = kernel(inMatrix, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols, fillGaps=fillGaps)
    weightedKernel = matrix.multiplyMatrices(kernelled, kernelMatrix.flatten().transpose())

    if fillGaps:
        new_sizeRows = inputs.size()[0] - kernel_size_rows + 1
        new_sizeCols = inputs.size()[1] - kernel_size_cols + 1
    else:
        new_sizeRows = ceil((inputs.size()[0] - kernel_size_rows + 1) / step_size_rows)
        new_sizeCols = ceil((inputs.size()[1] - kernel_size_cols + 1) / step_size_cols)

    return weightedKernel.reshape(new_sizeRows, new_sizeCols)
    # Where else has there been this problem with the reshape where it was flattening the object?

class Convolutional:
    def __init__(self, weight_set, step_size_rows, step_size_cols, activation_func):
        self.weights = weight_set
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

    def predict(self, inputs, training=False):
        # Test how the zero creation works

        if (training):
            out = weightedKernel(inputs, self.weights, self.step_size_rows, self.step_size_cols, fillGaps=True)
        else:
            out = weightedKernel(inputs, self.weights, self.step_size_rows, self.step_size_cols, fillGaps=False)

        outCpy = out.clone() # This is required for the softmax function
        out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        return out

    def train(self, input_set_raw, predicted, errors_raw, optimizer, learn_rate=0.5):
        self.iteration += 1

        errors = applyActivationGradient(self.activation_func, errors_raw, predicted)

        input_set = kernel(input_set_raw, self.kernel_size_rows, self.kernel_size_cols, self.step_size_rows, self.step_size_cols)
        input_set_transposed = input_set.transpose()

        w_AdjustmentsRaw = matrix.multiplyMatrices(input_set_transposed, errors.flatten())

        self.pWeight, self.rmsWeight, w_Adjustments = optimizer(self.pWeight, self.rmsWeight, w_AdjustmentsRaw, self.iteration)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate)
        self.weights = matrix.subtract(self.weights, w_Adjustments)

        # I have to make sure that the way it applies the gradients makes it so that it actually does the kernel layers correctly
        errorsShaped = errors.reshape(errors_raw.shape()[0], errors_raw.shape()[1])
        weightsFlipped = self.weights.rotate()
        errorsPadded = errorsShaped.pad(pad_up=weightsFlipped.size()[1]-1, pad_down=weightsFlipped.size()[1]-1, pad_left=weightsFlipped.size()[0]-1, pad_right=weightsFlipped.size()[0]-1)

        h_Error = weightedKernel(errorsPadded, weightsFlipped, 1, 1)
        return h_Error

        # I might make a seperate layer which handles the prediction layer and then do some modifiying to it

    def returnNetwork(self):
        pass

from misc import relu
weights = matrix.Matrix(dims=[2, 2], init=lambda: 0.5)
inputs = matrix.Matrix(dims=[3, 4], init=lambda: 2)
x = Convolutional(weights, 2, 2, relu)
x.predict(inputs).print()
print()
x.predict(inputs, training=True).print()
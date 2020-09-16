import matrix
from misc import backErrors
from math import ceil

def kernel(inMatrix, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols):
    # Do I need to have seperate kernelling size for when we kernel things that are not the weights such as the errors

    # The kernel size is actually gonna be the size of the weights
    size = inMatrix.size()

    retMatrix = []
    for rowNum in range(size[0]-kernel_size_rows+1):
        for colNum in range(size[1]-kernel_size_cols+1):
            if ((rowNum % step_size_rows == 0) and (colNum % step_size_cols == 0)):
                tempTrix = inMatrix.cut(rowNum, rowNum+kernel_size_rows, colNum, colNum+kernel_size_cols).flatten().returnMatrix()[0]
                retMatrix.append(tempTrix)

    return matrix.Matrix(arr=retMatrix)

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

    def predict(self, inputs, custom_filter=False):
        slider = custom_filter

        # Might need to add custom step size and row size for this or maybe not
        if not custom_filter:
            slider = self.weights

        kernels = kernel(inputs, self.kernel_size_rows, self.kernel_size_cols, self.step_size_rows, self.step_size_cols) 

        out = matrix.multiplyMatrices(kernels, slider.flatten().transpose())

        outCpy = out.clone() # This is required for the softmax function
        out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        new_sizeRows = ceil((inputs.size()[0] - slider.size()[0] + 1) / self.step_size_rows)
        new_sizeCols = ceil((inputs.size()[1] - slider.size()[1] + 1) / self.step_size_cols)

        out = out.flatten().reshape(new_sizeRows, new_sizeCols)

        return out

    def train(self, input_set_raw, predicted, errors, optimizer, learn_rate):
        self.iteration += 1

        # Thisapplies the multiplication of the derivative of the activation function
        # Will this need to be transposed?
        errors = backErrors(self.activation_func, errors, predicted)

        # The input layer will just be the same kernelled values from the other layer
        kernelTranspose = kernel(input_set_raw, self.kernel_size_rows, self.kernel_size_cols, self.step_size_rows, self.step_size_cols).transpose()

        # Now here we do the error creation for the actual errors
        # Note that the errors parsed in should be a flattened layer
        w_AdjustmentsRaw = matrix.multiplyMatrices(kernelTranspose, errors)

        self.pWeight, self.rmsWeight, w_Adjustments = optimizer(self.pWeight, self.rmsWeight, w_AdjustmentsRaw, self.iteration)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate)
        self.weights = matrix.subtract(self.weights, w_Adjustments)

        # Now for calculating the hidden errors
        # I need to pad that layer and then return a rotated kernel matrix
        # How does this kernelling for the hidden errors and the errors take affect when the step size is increased
        filterFlipped = self.weights.rotate()

        errorShape = predicted.size()
        filterShape = filterFlipped.shape()

        errorsShaped = errors.reshape(errorShape[0], errorShape[1])
        errorsPadded = errorsShaped.pad(pad_up=filterShape[1]-1, pad_down=filterShape[1]-1, pad_left=filterShape[0]-1, pad_right=filterShape[0]-1)

        # Now to get the errors for the previous layer I need to pad the error layer reshaped and then the 
        h_Error = self.predict(errorsPadded, filterFlipped)
        return h_Error

    def returnNetwork(self):
        pass

from misc import relu
weights = matrix.Matrix(dims=[2, 2], init=lambda: 4)
inputs = matrix.Matrix(dims=[10, 10], init=lambda: 2)
x = Convolutional(weights, 1, 1, relu)
x.predict(inputs).print()
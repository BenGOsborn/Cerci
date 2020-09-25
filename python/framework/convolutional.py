import matrix
import tensor
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

class Conv2d:
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

    def predict(self, inputs, applyActivation=True):
        wKernel = weightedKernel(inputs, self.weights, self.step_size_rows, self.step_size_cols)
        biasSet = matrix.Matrix(dims=wKernel.size(), init=lambda: self.bias)

        out = matrix.add(wKernel, biasSet)

        if (applyActivation):
            outCpy = out.clone()
            out = out.applyFunc(lambda x: self.activation_func(x, vals=outCpy))

        return out

    def train(self, input_set, errors_raw, optimizer, predicted=None, applyActivation=True, learn_rate=0.1):
        self.iteration += 1

        if (applyActivation):
            errors = applyActivationGradient(self.activation_func, errors_raw, predicted).transpose()
        else:
            errors = errors_raw

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

class ConvBlock:
    def __init__(self, weightTensor, biasMatrix, step_size_rows, step_size_cols, activation_func):
        if (weightTensor.size()[-1] != biasMatrix.size()[-1]): raise Exception(f"Tensor depths of weights and biases do not line up: Weight depth: {weightTensor.size()[-1]} | Bias depth: {biasMatrix.size()[-1]}")

        self.activation_func = activation_func

        self.convNets = []
        matrices = zip(weightTensor.returnTensor(), biasMatrix.flatten().returnMatrix()[0])
        for weights, bias in matrices:
            net = Conv2d(weights, bias, step_size_rows, step_size_cols, activation_func)
            self.convNets.append(net)

    def predict(self, inputTensor):
        if inputTensor.size()[2] != len(self.convNets): raise Exception(f"Tensor depths are not the same! Input depth: {inputTensor.size()[2]} | Kernel tensor depth: {len(self.convNets)}")
        inTensorsRaw = inputTensor.returnTensor()

        tensorRaw = []
        for matrix, net in zip(inTensorsRaw, self.convNets):
            out = net.predict(matrix, applyActivation=False)
            tensorRaw.append(out)

        outRaw = tensor.tensorSum(tensor.Tensor(tensorRaw))
        outCpy = outRaw.clone()
        outActivation = outRaw.applyFunc(lambda x: self.activation_func(x, vals=outCpy))
    
        return outActivation

    def train(self, inputTensor, predicted, errors_raw, optimizer, learn_rate=0.1):
        errors = applyActivationGradient(self.activation_func, errors_raw, predicted).transpose()

        backErrors = []
        for inputs, cnns in zip(inputTensor.returnTensor(), self.convNets):
            backError = cnns.train(inputs, errors, optimizer, applyActivation=False, learn_rate=learn_rate)
            backErrors.append(backError)

        return tensor.Tensor(backErrors)

# How does the output shape affect the rest of the network, I need to do some sort of test convolution for this one
# I also need to check where the activation function gets applied so we can take the derivatives properly
class Conv:
    def __init__(self, weightsFilters, biasFilterTensor, step_size_rows, step_size_cols, activation_func):
        if (weightsFilters.size()[-1] != biasFilterTensor.size()[-1]): raise Exception(f"Tensor numbers are not the same! Weights numbers: {weightsFilters.size()[-1]} | Bias numbers: {biasFilterTensor.size()[-1]}")

        self.activation_func = activation_func

        self.convNets = []
        for weights, bias in zip(weightsFilters.returnFilters(), biasFilterTensor.returnTensor()):
            cnnBlock = ConvBlock(weights, bias, step_size_rows, step_size_cols, activation_func)
            self.convNets.append(cnnBlock)

    def predict(self, inputTensor):
        outputs = []
        for net in self.convNets:
            predictionMatrix = net.predict(inputTensor) 
            outputs.append(predictionMatrix)

        return tensor.Tensor(outputs)

    def train(self, inputTensor, predictedTensor, errors_rawTensor, optimizer, learn_rate=0.1):

        hiddenPrev = []
        for errorsRaw, predicted, cnn in zip(self.convNets, predictedTensor.returnTensor(), errorsRaw.returnTensor()):
            hidden = cnn.train(inputTensor, predicted, errorsRaw, optimizer, learn_rate=learn_rate)
            hiddenPrev.append(hidden)

        return tensor.Tensor(hiddenPrev)

class Flatten:
    def __init__(self):
        # Now I'm going to have to keep some sort of reference for how the network will take in the values and store them for their appropriate values and into their appropriate tensors
        # For example if I flatten a specific kind of tensor then ti will do some wacky stuff

        # This means we will have to split the output matrix by the tensor length of each, and then for each split section we need to turn them into a big tensor

        # So now we want to carry through the shape of the input tensor parsed through

        self.__out_rows = None
        self.__out_cols = None
        self.__out_layers = None
        self.__input_tensor = None
        
    def flatten(self, inputTensor):
        self.__input_tensor = inputTensor
        self.__out_rows, self.__out_cols, self.__out_layers = inputTensor.size()

        out = []
        for mat in inputTensor.returnTensor():
            flatMat = mat.flatten().returnMatrix()[0] 
            for val in flatMat:
                out.append(val)

        return matrix.Matrix(out)

    def reshapeErrors(self, flatErrors):
        flatErrorRaw = flatErrors.returnMatrix()[0]

        size = self.__out_rows * self.__out_cols
        matrices = []
        for i in range(self.__out_layers):
            matRaw = flatErrorRaw[i*size:(i+1)*size]
            mat = matrix.Matrix(arr=matRaw)
            matReshaped = mat.reshape(self.__out_rows, self.__out_cols)
            matrices.append(matReshaped)

        return tensor.Tensor(matrices)

class Pool:
    def __init__(self, kernel_size_rows, kernel_size_cols, step_size_rows, step_size_cols):
        self.__kernel_size_rows = kernel_size_rows
        self.__kernel_size_cols = kernel_size_cols
        self.__step_size_rows = step_size_rows
        self.__step_size_cols = step_size_cols
        self.__tensorIndexes = []

    def __maxMatrix(self, inMatrix):
        flat = inMatrix.flatten().returnMatrix()[0]

        mx = 0
        dex = 0
        for i, val in enumerate(flat):
            if val > mx:
                mx = val
                dex = i

        row = 0
        while (dex > 0):
            dex-inMatrix.size()[1]
            row += 1

        return mx, row, dex

    def __poolMatrix(self, inMatrix):
        size = inMatrix.size()

        retMatrix = []
        storeMatrix = []
        for rowNum in range(size[0]-self.__kernel_size_rows+1):
            tempRowRet = []
            tempRowStore = []
            for colNum in range(size[1]-self.__kernel_size_cols+1):
                if ((rowNum % self.__step_size_rows == 0) and (colNum % self.__step_size_cols == 0)):
                    tempTrix = inMatrix.cut(rowNum, rowNum+self.__kernel_size_rows, colNum, colNum+self.__kernel_size_cols)
                    mx, relRow, relCol = self.__maxMatrix(tempTrix)
                    tempRowRet.append(mx)

                    # Am I going to need the mx too for this one, or just keep all of this?
                    # This will give us the coord points for where the error corresponds to so we can add it up correctly
                    tempRowStore.append([rowNum+relRow, colNum+relCol])

            retMatrix.append(tempRowRet)
            storeMatrix.append(tempRowStore)

        matStoreMatrix = matrix.Matrix(arr=storeMatrix)
        self.__tensorIndexes.append(matStoreMatrix)
        return matrix.Matrix(arr=storeMatrix)

    def pool(self, inputTensor):
        self.__tensorIndexes = []
        tensorRaw = inputTensor.returnTensor()

        tempTensor = []
        for mat in tensorRaw:
            pooledMatrix = self.__poolMatrix(mat) 
            tempTensor.append(pooledMatrix)

        # So now that I have the data for the array which is not getting stored how do I extract this information for the error layers
        # Or more like how do I store these values appropriately andparse them through

        return tensor.Tensor(tempTensor)

    def reshapeErrors(self, unshapedErrors):
        pass
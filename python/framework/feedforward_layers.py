import matrix
# import misc

class FeedForward:

    def __init__(self, weight_set, bias_set, activation_func, beta1=0.9, beta2=0.999, epsilon=10e-8):
        self.weights = weight_set
        self.bias = bias_set
        self.activation_func = activation_func

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.pWeight = matrix.Matrix(dims=weight_set.size(), init=0)
        self.pBias = matrix.Matrix(dims=bias_set.size(), init=0)
        self.rmsWeight = matrix.Matrix(dims=weight_set.size(), init=0)
        self.rmsBias = matrix.Matrix(dims=bias_set.size(), init=0)
        self.iteration = 0 # This has to be reinited at the start of every training session

    def feedForward(self, inputs):
        multiplied = matrix.multiplyMatrices(self.weights, inputs)
        out = matrix.add(multiplied, self.bias)

        outCpy = matrix.Matrix(out.returnMatrix())
        out.applyFunc(lambda x: self.activation_func(x, vals=outCpy)) # Done for consistency reasons when there is need for a 'deriv=True' parsed through

        return out

    # This clears the momentum buffer when new sets need to be trained on the model
    def reinit(self):
        self.pWeight = matrix.Matrix(dims=self.weights.size(), init=0)
        self.pBias = matrix.Matrix(dims=self.bias.size(), init=0)
        self.rmsWeight = matrix.Matrix(dims=self.weights.size(), init=0)
        self.rmsBias = matrix.Matrix(dims=self.bias.size(), init=0)
        self.iteration = 0

    def train(self, input_set, loss_func, errors, optimizer, learn_rate=0.5):
        # I think I rather want to do the prediction at the start of the training, where I also gather the outputs for each layer which act as the inputs for each node, and then I have the predictions
        # This means we do the training set on the pre iteration and then lead with that as the errors
        # This also means we do the training error from the first layer on the pre iteration
        self.iteration += 1

        inputTransposed = matrix.Matrix(arr=input_set.returnMatrix())
        inputTransposed.transpose()

        w_AdjustmentsRaw = matrix.multiplyMatrices(errors, inputTransposed)

        self.pWeight, self.rmsWeight, w_Adjustments = optimizer(self.pWeight, self.rmsWeight, w_AdjustmentsRaw, self.beta1, self.beta2, self.epsilon, self.iteration)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate)
        w_New = matrix.subtract(self.weights, w_Adjustments)
        self.weights = w_New

        self.pBias, self.rmsBias, b_Adjustments = optimizer(self.pBias, self.rmsBias, errors, self.beta1, self.beta2, self.epsilon, self.iteration)
        b_Adjustments = matrix.multiplyScalar(errors, learn_rate)
        b_New = matrix.subtract(self.bias, b_Adjustments)
        self.bias = b_New

        transposeWeights = matrix.Matrix(arr=self.weights.returnMatrix())
        transposeWeights.transpose()

        h_Error = matrix.multiplyMatrices(transposeWeights, errors)
        return h_Error

    def getWeights(self):
        return self.weights, self.bias
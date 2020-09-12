import matrix
import misc

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

    def train(self, input_set, loss_func, training_set=False, hidden_errors=False):
        self.iteration += 1

        # Maybe ill ad an optimizer implementation in here to be able to switch optimizers based on the previous function

        predicted = self.feedForward(input_set)

        if (training_set != False):
            errors = misc.backErrors(loss_func, self.activation_func, predicted, training_set, predicted.size())       
        elif (hidden_errors != False):
            errors = hidden_errors

        # Implement adam here
        learn_rate = 0.5

        inputTransposed = matrix.Matrix(arr=input_set.returnMatrix())
        inputTransposed.transpose()

        w_AdjustmentsRaw = matrix.multiplyMatrices(errors, inputTransposed)

        w_Adjustments, self.pWeight, self.rmsWeight = misc.adam(self.pWeight, self.rmsWeight, w_AdjustmentsRaw, self.beta1, self.beta2, self.epsilon, self.iteration)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate)
        w_New = matrix.subtract(self.weights, w_Adjustments)
        self.weights = w_New

        b_Adjustments, self.pBias, self.rmsBias = misc.adam(self.pBias, self.rmsBias, errors, self.beta1, self.beta2, self.epsilon, self.iteration)
        b_Adjustments = matrix.multiplyScalar(errors, learn_rate)
        b_New = matrix.subtract(self.bias, b_Adjustments)
        self.bias = b_New

        transposeWeights = matrix.Matrix(arr=self.weights.returnMatrix())
        transposeWeights.transpose()

        h_Error = matrix.multiplyMatrices(transposeWeights, errors)
        return h_Error

    def getWeights(self):
        return self.weights, self.bias

weights = matrix.Matrix([[0.5, 0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5, 0.5]])
                         
bias = matrix.Matrix([[0.5],
                      [0.5],
                      [0.5]])

inputs = matrix.Matrix([[1],
                        [0],
                        [1],
                        [0]])

training = matrix.Matrix([[1],
                        [0],
                        [0]])

brain = FeedForward(weights, bias, misc.sigmoid)

for _ in range(9):
    brain.train(inputs, misc.crossEntropy, training_set=training)

brain.feedForward(inputs).print()
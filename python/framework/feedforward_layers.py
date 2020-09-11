import matrix
import misc

class FeedForward:

    def __init__(self, weight_set, bias_set, activation_func):
        self.weights = weight_set
        self.bias = bias_set
        self.activation_func = activation_func

    def feedForward(self, inputs):
        multiplied = matrix.multiplyMatrices(self.weights, inputs)
        out = matrix.add(multiplied, self.bias)

        outCpy = matrix.Matrix(out.returnMatrix())
        out.applyFunc(lambda x: self.activation_func(x, vals=outCpy)) # Done for consistency reasons when there is need for a 'deriv=True' parsed through

        return out

    def train(self, input_set, loss_func, training_set=False, hidden_errors=False):
        predicted = self.feedForward(input_set)

        # Implement momentum here
        if (training_set != False):
            errors = misc.backErrors(loss_func, self.activation_func, predicted, training_set, predicted.size())       
        elif (hidden_errors != False):
            errors = hidden_errors

        # Implement adam here
        learn_rate = 0.5

        inputTransposed = matrix.Matrix(arr=input_set.returnMatrix())
        inputTransposed.transpose()

        w_Adjustments = matrix.multiplyMatrices(errors, inputTransposed)
        w_Adjustments = matrix.multiplyScalar(w_Adjustments, learn_rate)
        w_New = matrix.subtract(self.weights, w_Adjustments)
        self.weights = w_New

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

training = matrix.Matrix([[0],
                        [1],
                        [0]])

brain = FeedForward(weights, bias, misc.softmax)

for _ in range(100):
    brain.train(inputs, misc.crossEntropy, training_set=training)

brain.feedForward(inputs).print()
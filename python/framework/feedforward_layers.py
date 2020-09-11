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

    def trainFinal(self, input_set, training_set, loss_func):
        predicted = self.feedForward(input_set)
        errors = loss_func(predicted, training_set)       

        

    def getWeights(self):
        return self.weights, self.bias

weights = matrix.Matrix([[0.5, 0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5, 0.5]])
                         
bias = matrix.Matrix([[0.5],
                      [0.5],
                      [0.5]])

inputs = matrix.Matrix([[0.5],
                        [0.5],
                        [0.5],
                        [0.5]])

brain = FeedForward(weights, bias, misc.softmax)
x = brain.feedForward(inputs)
x.print()
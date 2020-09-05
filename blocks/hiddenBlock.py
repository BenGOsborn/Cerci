from resources import sigmoid, dot

class HiddenBlock:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, hidden_inputs):
        pass

    def train(self, hidden_inputs, hidden_errors):
        pass
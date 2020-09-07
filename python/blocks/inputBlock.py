from resources import relu, learnFunc, dot

class InputBlock:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        output = [
            relu(
                dot(inputs, weights) + self.bias
            ) 
        for weights in self.weights]
        return output

    def train(self, input_data, hidden_errors):
        error = sum(hidden_errors) / len(hidden_errors)
        predictions = self.feedForward(input_data)

        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                update = error*relu(predictions[y], deriv=True)*input_data[x]
                learn_rate = learnFunc(update)
                self.weights[y][x] -= learn_rate*update

        biasUpdate = 0
        for x in range(len(self.weights)):
            biasUpdate += error*relu(predictions[x], deriv=True)/len(predictions)
        learn_rate = learnFunc(biasUpdate)
        self.bias -= learn_rate*biasUpdate

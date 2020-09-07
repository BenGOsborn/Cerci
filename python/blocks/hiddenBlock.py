from resources import relu, learnFunc, dot

class HiddenBlock:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, hidden_inputs):
        output = [
            relu(
                dot(hidden_inputs, weights) + bias
            ) 
        for weights, bias in zip(self.weights, self.bias)]
        return output

    def train(self, hidden_inputs, hidden_errors):
        error = sum(hidden_errors) / len(hidden_errors)
        predictions = self.feedForward(hidden_inputs)

        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                update = error*relu(predictions[y], deriv=True)*hidden_inputs[x]
                learn_rate = learnFunc(update)
                self.weights[y][x] -= learn_rate*update

        for x in range(len(self.weights)):
            update = error*relu(predictions[x], deriv=True)
            learn_rate = learnFunc(update)
            self.bias[x] -= learn_rate*update

        prevErrors = []
        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                prevError = error*relu(predictions[y], deriv=True)*self.weights[y][x]
                prevErrors.append(prevError)
        
        return prevErrors
from resources import sigmoid, dot

class OutputBlock:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, hidden_inputs):
        output = [
            sigmoid(
                dot(hidden_inputs, weights) + bias
            ) 
        for weights, bias in zip(self.weights, self.bias)]
        return output

    def train(self, hidden_inputs, training_data):
        error = 0
        for data, train_data in zip(hidden_inputs, training_data):
            error += data - train_data

        predictions = self.feedForward(hidden_inputs)

        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                update = error*sigmoid(predictions[y], deriv=True)*hidden_inputs[x]
                self.weights[y][x] -= 0.5*update

        for x in range(len(self.weights)):
            update = error*sigmoid(predictions[x], deriv=True)
            self.bias[x] -= 0.5*update

        prevErrors = []
        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                prevError = error*sigmoid(predictions[y], deriv=True)*self.weights[y][x]
                prevErrors.append(prevError)

        return prevErrors

# weights = [
#     [0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5]
# ]

# bias = [
#     0.5,
#     0.5,
#     0.5
#     ]

# x = OutputBlock(weights, bias)

# inputs = [0.5, 0.5, 0.5]

# print(x.feedForward(inputs))
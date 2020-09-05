from resources import sigmoid, dot

class InputBlock:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        output = [
            sigmoid(
                dot(inputs, weights) + bias
            ) 
        for weights, bias in zip(self.weights, self.bias)]
        return output

    def train(self, input_data, hidden_errors):
        # error = 0
        # for data, train_data in zip(input_data, hidden_errors):
        #     error += data - train_data

        error = sum(hidden_errors) / len(hidden_errors)
        predictions = self.feedForward(input_data)

        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                update = error*sigmoid(predictions[y], deriv=True)*input_data[x]
                self.weights[y][x] -= 0.5*update

        for x in range(len(self.weights)):
            update = error*sigmoid(predictions[x], deriv=True)
            self.bias[x] -= 0.5*update
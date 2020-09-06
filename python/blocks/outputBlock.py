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
        # This doesnt make much sense to have as there could be different values and this is silly
        
        predictions = self.feedForward(hidden_inputs)
        if (len(predictions) != len(training_data)): raise Exception(f"Predictions length is not same length as train data! Predictions Length: {len(predictions)} | Data Length: {len(training_data)}")

        for prediction, train_data in zip(predictions, training_data):
            error += prediction - train_data

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
# ]

# bias = [
#     0.5,
#     ]

# x = OutputBlock(weights, bias)

# inputs = [1, 0, 0]
# training_data = [0]

# print(x.feedForward(inputs))
# x.train(inputs, training_data)
# print(x.feedForward(inputs))
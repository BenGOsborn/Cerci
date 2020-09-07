from resources import relu, sigmoid, learnFunc, dot

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
        predictions = self.feedForward(hidden_inputs)
        if (len(predictions) != len(training_data)): raise Exception(f"Predictions length is not same length as train data! Predictions Length: {len(predictions)} | Data Length: {len(training_data)}")

        error = 0
        for prediction, train_data in zip(predictions, training_data):
            error += (prediction - train_data)/len(predictions)

        # So instead of the learning rate, take the learning rate to be a sigmoid of the absolute value of how its degressing, it should not change the sign

        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                update = error*sigmoid(predictions[y], deriv=True)*hidden_inputs[x]
                learn_rate = learnFunc(update)
                self.weights[y][x] -= learn_rate*update

        for x in range(len(self.weights)):
            update = error*sigmoid(predictions[x], deriv=True)
            learn_rate = learnFunc(update)
            self.bias[x] -= learn_rate*update

        prevErrors = []
        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                prevError = error*sigmoid(predictions[y], deriv=True)*self.weights[y][x]
                prevErrors.append(prevError)

        return prevErrors
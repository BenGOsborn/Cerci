from resources import relu, sigmoid, learnFunc, dot

class OutputBlock:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, hidden_inputs):
        output = [
            sigmoid(
                dot(hidden_inputs, weights) + self.bias
            ) 
        for weights in self.weights]

        return output

    def train(self, hidden_inputs, training_data):
        predictions = self.feedForward(hidden_inputs)
        if (len(predictions) != len(training_data)): raise Exception(f"Predictions length is not same length as train data! Predictions Length: {len(predictions)} | Data Length: {len(training_data)}")

        error = 0
        for prediction, train_data in zip(predictions, training_data):
            error += (prediction - train_data)/len(predictions)

        # So instead of the learning rate, take the learning rate to be a sigmoid of the absolute value of how its degressing, it should not change the sign
        prevErrors = []
        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                prevError = error*sigmoid(predictions[y], deriv=True)*self.weights[y][x]
                prevErrors.append(prevError)

        for y in range(len(self.weights)):
            for x in range(len(self.weights[0])):

                # We want to check the last layer only and have a deep look at the values which are being contributed to it

                update = error*sigmoid(predictions[y], deriv=True)*hidden_inputs[x]
                learn_rate = learnFunc(update)
                self.weights[y][x] -= learn_rate*update

                # This needs adjusting so that it activates at the start of when the error occurs
                if ((y == len(self.weights)-1) and (sum(self.weights[y]) < 1)):
                        print(f"Weight update: {update} | Adjusted weights: {self.weights[y]} | Predictions: {predictions} | Error: {error}")

        biasUpdate = 0
        for x in range(len(self.weights)):
            biasUpdate += error*sigmoid(predictions[x], deriv=True)/len(predictions)
        learn_rate = learnFunc(biasUpdate)
        self.bias -= learn_rate*biasUpdate

        return prevErrors
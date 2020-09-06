from blocks.resources import sigmoid, dot

class Brain:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        raw = dot(inputs, self.weights) + self.bias
        return sigmoid(raw)

    def trainNetwork(self, input_data, training_data):
        # 1/2n*sum(predicted-observed)^2 - One half mean squared error
        predicted = self.feedForward(input_data)

        error = (predicted - training_data) 

        for i in range(len(self.weights)):
            update = error*sigmoid(predicted, deriv=True)*input_data[i]
            self.weights[i] -= 0.5*update

        biasUpdate = error*sigmoid(predicted, deriv=True)
        self.bias -= 0.5*biasUpdate

weights = [0.5, 0.5, 0.5]
bias = 0.5

inputs = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]
training_data = [0, 0, 0, 1, 1, 0, 1, 1]

brain = Brain(weights, bias)

for _ in range(10000):
    for i in range(len(inputs)):
        brain.trainNetwork(inputs[i], training_data[i])

print(f"Weights: {brain.weights} Biases: {brain.bias}")
error = 0
for i in range(len(inputs)):
    error += (training_data[i] - brain.feedForward(inputs[i]))/len(inputs) 
    print(error)
print(f"Error: {error}")
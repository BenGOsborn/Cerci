import resources

# Am I going to want to set up my own classes for this such as their own objects?
# Yes I will need to have seperate modules for this

# This is going to be the input layer for the network, we want to be working in modules here
class InputLayer:
    def __init__(self, weights, bias):
        # These weights will be parsed as an array of weights and biases
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        output = [
            resources.sigmoid(
                resources.dot(inputs, weights) + bias
            ) 
        for weights, bias in zip(self.weights, self.bias)]
        return output

    def train(self, hidden_errors):
        # We want to calculate the error as a big sum
        # We want to iterate over the weight layers aswell as the layers in the individual networks themselves which come from the network

        # Now is this hidden errors result going to be accurate or not and is there going to be a better way of doing this?
        error = sum(hidden_errors)/len(hidden_errors)
        # Am I going to have to do a comparison between the error values here aswell?

        # This is going to deal with the weights section

        # The y is going to be dealing with the hidden error values
        for y in range(len(self.weights)):
            # The x is going to be iterating over the actual values in the weights
            for x in range(len(self.weights[0])):
                
                pass

        # This is going to deal with the bias section

        pass

class HiddenLayer:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, hidden_inputs):
        pass

    def train(self, hidden_inputs, hidden_errors):
        pass

class OutputLayer:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, hidden_inputs):
        pass

    def train(self, hidden_inputs, training_data):
        pass

# Construct the neural network manually down here
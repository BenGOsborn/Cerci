from inputBlock import InputBlock
from hiddenBlock import HiddenBlock
from outputBlock import OutputBlock

class Brain:

    def __init__(self, weights, bias):
        self.networkLayers = []
        # How am I defining the size of these layers though?

        if len(weights) == 0:
            raise Exception("Network requires atleast one layer")

        elif len(weights) == 1:
            print("Length of 1")
            print(weights, bias)

            outLayer = OutputBlock(weights, bias) 
            self.networkLayers.append(outLayer)

        elif len(weights) == 2:
            print("Length of 2")
            print(weights[1], bias[1])

            inLayer = InputBlock(weights[0], bias[0])
            self.networkLayers.append(inLayer)

            outLayer = OutputBlock(weights[1], bias[1])
            self.networkLayers.append(outLayer)

        else:
            inLayer = InputBlock(weights[0], bias[0])
            self.networkLayers.append(inLayer)

            for i in range(1, len(weights) - 1):
                hiddenLayer = HiddenBlock(weights[i], bias[i])
                self.networkLayers.append(hiddenLayer)

            outLayer = OutputBlock(weights[-1], bias[-1])
            self.networkLayers.append(outLayer)
    
    def feedForward(self, inputs):
        outputs = inputs
        for layer in self.networkLayers:
            outputs = layer.feedForward(outputs)

        return outputs

    # This is just going to be a basic single input single training data
    # I will write a higher level data parser afterwards
    def train(self, input_data, training_data):
        predicted_array = []

        # We should add the input data to this to make it make more sense
        feed_values = input_data
        for layer in self.networkLayers:
            feed_values = layer.feedForward(feed_values)
            predicted_array.append(feed_values)

        # So now it has all of the layers, and we have to go backwards now
        reversed_layers = self.networkLayers[::-1]

        # This function has not been finished yet

# I want to add a feature where it autogens the weights by default if given an array of weights instead of nums by having random values
# The size of the brain will be defined through the size of the weights
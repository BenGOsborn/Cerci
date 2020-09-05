from blocks.inputBlock import InputBlock
from blocks.hiddenBlock import HiddenBlock
from blocks.outputBlock import OutputBlock

class Brain:

    # Think about the architecture
    # If there is one layer then it will simply just do an output layer
    # If there is more than one layer then there is going to have to be an input layer
    # For everything else there is going to have to be hidden layers inbetween 

    # Im going to need weight and bias parsing for this took
    def __init__(self, layers, weights, bias):
        self.networkLayers = []

        if len(layers) == 0:
            raise Exception("Network requires atleast one layer")
        elif len(layers) == 1:
           # This means its gotta be parsed as just a single weight and bias value to work here
           outLayer = OutputBlock(weights, bias) 
           self.networkLayers.append(outLayer)
        elif len(layers) == 2:
            inLayer = InputBlock(weights[0], bias[0])
            outLayer = OutputBlock(weights[1], bias[1])
            self.networkLayers.append(inLayer)
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
        for i in range(len(self.networkLayers)):
            outputs = self.networkLayers[i].feedForward(outputs)

        return outputs

    # This is just going to be a basic single input single training data
    # I will write a higher level data parser afterwards
    def train(self, input_data, training_data):
        pass

# Make valid training data for this to test it
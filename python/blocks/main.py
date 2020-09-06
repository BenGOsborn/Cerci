from inputBlock import InputBlock
from hiddenBlock import HiddenBlock
from outputBlock import OutputBlock

from resources import trainData

class Brain:

    def __init__(self, weights, bias):
        self.networkLayers = []
        # How am I defining the size of these layers though?

        if len(weights) == 0:
            raise Exception("Network requires atleast one layer")

        elif len(weights) == 1:
            outLayer = OutputBlock(weights[0], bias[0]) 
            self.networkLayers.append(outLayer)

        elif len(weights) == 2:
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
        prev_fed_through = [input_data]

        # We should add the input data to this to make it make more sense
        feed_values = input_data
        for layer in self.networkLayers[:-1]:
            feed_values = layer.feedForward(feed_values)
            prev_fed_through.append(feed_values)

        # So now it has all of the layers, and we have to go backwards now
        reversed_layers = self.networkLayers[::-1]

        # Pseudo:

        # First part
        # For the first in the array which is the final output layer in the reversed layers
        # We update this layers weights using the training data and the prev_fed_through last value
        # We then grab these error values and then set that as the prev error, and then feed it into the previous layer (This means we should start with the training_data in that fedthrougherror)
        # We then iterate over the whole array until we get to the end of the layers, and the last one SHOULD be finishing with the 'input_data'

# All items have a standard 3 length input
items = trainData()

brain = Brain(items['weightsMulti'], items['biasMulti'])

inputs = [0.5, 0.5, 0.5]
actual = [0.5, 0.5, 0.5]

print(brain.train(inputs, actual))
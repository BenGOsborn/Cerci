from inputBlock import InputBlock
from hiddenBlock import HiddenBlock
from outputBlock import OutputBlock

from resources import trainData, error

class Brain:

    def __init__(self, weights, bias):
        self.networkLayers = []

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
    
    def getModel(self):
        # Make this spit out an array instead
        for layer in self.networkLayers:
            print(layer.weights)
            print(layer.bias)
            print()

    def feedForward(self, inputs):
        outputs = inputs
        for layer in self.networkLayers:
            outputs = layer.feedForward(outputs)

        return outputs

    # This is just going to be a basic single input single training data
    # I will write a higher level data parser afterwards
    def train(self, input_data, training_data):
        prev_outputs = [input_data]

        # We should add the input data to this to make it make more sense
        outputs = input_data
        for layer in self.networkLayers[:-1]:
            outputs = layer.feedForward(outputs)
            prev_outputs.append(outputs)

        # Pseudo:

        # First part
        # For the first in the array which is the final output layer in the reversed layers
        # We update this layers weights using the training data and the prev_fed_through last value
        # We then grab these error values and then set that as the prev error, and then feed it into the previous layer (This means we should start with the training_data in that fedthrougherror)

        # This means it should start off with the previous inputs

        # It is not parsing through the correct values and I am not sure why not?
        error_values = training_data
        for layer in self.networkLayers[::-1]:
            hidden_input = prev_outputs.pop(-1)
            error_values = layer.train(hidden_input, error_values)

# The model isnt even making accurate predictions for the layer where it should be (outputBlock)

# All items have a standard 3 length input

inputs = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]

actual = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

items = trainData()

mode = "Multi"
act_weights, act_bias = items[f"weights{mode}"], items[f"bias{mode}"]

brain = Brain(act_weights, act_bias)

for _ in range(10000):
    for inp, act in zip(inputs, actual):
        brain.train(inp, act)

err = 0
for inp, act in zip(inputs, actual):
    vals = brain.feedForward(inp)
    # vals = [round(val) for val in vals]
    print(f"Predicted values: {vals} | Actual values: {act}")

    err += error(vals, act)
print(f"Error: {err}")

# Maybe the taking of the averages is too much?
# Its approximating the values too hard and I dont know why...
# Might have to instead of doing this take the actual error values for the first one

# It shouldnt nof been wrong because it was going to the node individually and not the array value
# Should it be fitting it to the network better because it is doing multiple epochs

# Maybe I should do line by line propogation again
# Add adjustments for the learning rate

# Still broken for a multi layer network, possibly because of just two many values to fit
# Test it with longer neuron amounts aswell

# Create a test system where the amount of layers can be created from the neurons
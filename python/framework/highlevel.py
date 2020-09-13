import feedforward_layers as ff
import misc
from matrix import Matrix, subtract

class Brain:
    # Each layer is the layer type, weights for the layer, bias for the layer, activation function for the layer
    # This does not take into consideration the beta1, beta2 and epsilon params for now
    def __init__(self, *layers):
        self.__layers = []

        for layer in layers:
            self.__layers.append(layer[0](layer[1], layer[2], layer[3], dropout_rate=layer[4]))

    def predict(self, inputs):
        feed = inputs
        for layer in self.__layers:
            feed = layer.feedForward(feed)
        return feed

    def train(self, input_set, training_set, loss_func=misc.crossEntropy, optimizer=misc.adam, learn_rate=0.5):
        hiddens = [input_set]
        feed = input_set
        for layer in self.__layers:
            feed = layer.feedForward(feed, training=True)
            hiddens.append(feed)

        l_Reversed = self.__layers[::-1]
        h_Reversed = hiddens[::-1]

        errors = misc.getDifferences(loss_func, feed, training_set)
        for i, layer in enumerate(l_Reversed):
            errors = layer.train(h_Reversed[i+1], h_Reversed[i], errors, optimizer, learn_rate=learn_rate)

    def returnModel(self):
        returnArray = []
        for layer in self.__layers:
            returnArray.append(layer.returnNetwork())
        return returnArray

weights1 = Matrix(dims=[2, 2], init="random")
bias1 = Matrix(dims=[2, 1], init="random")
weights2 = Matrix(dims=[2, 2], init="random")
bias2 = Matrix(dims=[2, 1], init="random")

inputs1 = Matrix(arr=[[1, 0]])
inputs1.transpose()
training1 = Matrix(arr=[[1, 0]])
training1.transpose()
inputs2 = Matrix(arr=[[0, 1]])
inputs2.transpose()
training2 = Matrix(arr=[[1, 0]])
training2.transpose()
inputs3 = Matrix(arr=[[1, 1]])
inputs3.transpose()
training3 = Matrix(arr=[[0, 1]])
training3.transpose()
inputs4 = Matrix(arr=[[0, 0]])
inputs4.transpose()
training4 = Matrix(arr=[[1, 0]])
training4.transpose()

brain = Brain(
    [ff.FeedForward, weights1, bias1, misc.relu, 0],
    [ff.FeedForward, weights2, bias2, misc.softmax, 0]
)
for _ in range(100):
    brain.train(inputs1, training1, loss_func=misc.crossEntropy)
    brain.train(inputs2, training2, loss_func=misc.crossEntropy)
    brain.train(inputs3, training3, loss_func=misc.crossEntropy)
    brain.train(inputs4, training4, loss_func=misc.crossEntropy)

prediction = brain.predict(inputs3)
prediction.transpose()
prediction.print()
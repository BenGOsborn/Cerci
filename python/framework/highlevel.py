import feedforward_layers as ff
import misc
from matrix import Matrix, subtract

class Brain:
    # Each layer is the layer type, weights for the layer, bias for the layer, activation function for the layer
    # This does not take into consideration the beta1, beta2 and epsilon params for now
    def __init__(self, *layers):
        self.__layers = []

        for layer in layers:
            self.__layers.append(layer[0](layer[1], layer[2], layer[3]))

    def predict(self, inputs, dropout_rate=0):
        feed = inputs
        for layer in self.__layers:
            feed = layer.feedForward(feed, dropout_rate)
        return feed

    def train(self, input_set, training_set, loss_func, optimizer, learn_rate, dropout_rate):
        hiddens = []
        feed = input_set
        for layer in self.__layers:
            feed = layer.feedForward(feed, dropout_rate=dropout_rate)
            hiddens.append(feed)

        l_Reversed = self.__layers[::-1]
        h_Reversed = hiddens[::-1]

        errors = misc.getDifferences(loss_func, feed, training_set)
        for i, layer in enumerate(l_Reversed):
            errors = layer.train(h_Reversed[i-1], h_Reversed[i], errors, optimizer, learn_rate=learn_rate)

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
training2 = Matrix(arr=[[0, 1]])
training2.transpose()
inputs3 = Matrix(arr=[[1, 1]])
inputs3.transpose()
training3 = Matrix(arr=[[1, 1]])
training3.transpose()
inputs4 = Matrix(arr=[[0, 0]])
inputs4.transpose()
training4 = Matrix(arr=[[0, 0]])
training4.transpose()

layer1 = ff.FeedForward(weights1, bias1, misc.relu)
layer2 = ff.FeedForward(weights2, bias2, misc.sigmoid)

for _ in range(15):
    prediction1 = layer1.feedForward(inputs1)
    prediction2 = layer2.feedForward(prediction1)
    errors = misc.getDifferences(misc.crossEntropy, prediction2, training1)
    errors = layer2.train(prediction1, prediction2, errors, misc.adam, learn_rate=0.5)
    layer1.train(inputs1, prediction1, errors, misc.adam, learn_rate=0.5)

    prediction1 = layer1.feedForward(inputs2)
    prediction2 = layer2.feedForward(prediction1)
    errors = misc.getDifferences(misc.crossEntropy, prediction2, training2)
    errors = layer2.train(prediction1, prediction2, errors, misc.adam, learn_rate=0.5)
    layer1.train(inputs2, prediction1, errors, misc.adam, learn_rate=0.5)

    prediction1 = layer1.feedForward(inputs3)
    prediction2 = layer2.feedForward(prediction1)
    errors = misc.getDifferences(misc.crossEntropy, prediction2, training3)
    errors = layer2.train(prediction1, prediction2, errors, misc.adam, learn_rate=0.5)
    layer1.train(inputs3, prediction1, errors, misc.adam, learn_rate=0.5)

    prediction1 = layer1.feedForward(inputs4)
    prediction2 = layer2.feedForward(prediction1)
    errors = misc.getDifferences(misc.crossEntropy, prediction2, training4)
    errors = layer2.train(prediction1, prediction2, errors, misc.adam, learn_rate=0.5)
    layer1.train(inputs4, prediction1, errors, misc.adam, learn_rate=0.5)

feed = layer1.feedForward(inputs4)
layer2.feedForward(feed).print()
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

weights1 = Matrix(dims=[4, 4], init=0.5)
bias1 = Matrix(dims=[4, 1], init=0.5)

weights2 = Matrix(dims=[3, 4], init=0.5)
bias2 = Matrix(dims=[3, 1], init=0.5)

brain = Brain([ff.FeedForward, weights1, bias1, misc.relu], 
              [ff.FeedForward, weights2, bias2, misc.softmax])

inputs = Matrix(arr=[[1, 1, 1, 1],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0]])

training = Matrix(arr=[[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

input_raw = inputs.returnMatrix()
training_raw = training.returnMatrix()

# Something has gone very wrong here when we try to train it for multiple values, why?

for _ in range(100):
    for i in range(len(input_raw)):
        put, train = Matrix(arr=input_raw[i]), Matrix(arr=training_raw[i])
        put.transpose()
        train.transpose()

        brain.train(put, train, misc.meanSquared, misc.adam, 0.05, 10000)

test_input = Matrix(arr=inputs.returnMatrix()[0])
test_input.transpose()
brain.predict(test_input).print()

# I think I need to do a big test based on a first layer and then that way I can check to see if something else is going wrong
import fullyconnected as fc
import misc
from matrix import Matrix, subtract

class Brain:
    def __init__(self, *layers):
        self.__layers = []

        for layer in layers:
            self.__layers.append(layer[0](layer[1], layer[2], layer[3]))

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



weights1 = Matrix(dims=[1, 2], init=lambda: 0.5)
bias1 = Matrix(dims=[1, 1], init=lambda: 0.5)

inputs1 = Matrix(arr=[[1, 0]]).transpose()
training1 = Matrix(arr=[[0]]).transpose()
inputs2 = Matrix(arr=[[0, 1]]).transpose()
training2 = Matrix(arr=[[0]]).transpose()
inputs3 = Matrix(arr=[[1, 1]]).transpose()
training3 = Matrix(arr=[[1]]).transpose()
inputs4 = Matrix(arr=[[0, 0]]).transpose()
training4 = Matrix(arr=[[0]]).transpose()

brain = Brain(
    [fc.FullyConnected, weights1, bias1, misc.sigmoid],
)

for _ in range(1000):
    brain.train(inputs1, training1, loss_func=misc.crossEntropy)
    brain.train(inputs2, training2, loss_func=misc.crossEntropy)
    brain.train(inputs3, training3, loss_func=misc.crossEntropy)
    brain.train(inputs4, training4, loss_func=misc.crossEntropy)

s = subtract(brain.predict(inputs1), training1).returnMatrix()[0][0] + subtract(brain.predict(inputs2), training2).returnMatrix()[0][0] + subtract(brain.predict(inputs3), training3).returnMatrix()[0][0] + subtract(brain.predict(inputs4), training4).returnMatrix()[0][0]
print(s)
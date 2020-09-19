import fullyconnected
import convolutional
import misc
import matrix
import mnist_parser

from random import random

data_set = mnist_parser.loadData("python/framework/data.pickle")

weights1 = matrix.Matrix(dims=[5, 5], init=lambda: 0.01)
bias1 = random()
weights2 = matrix.Matrix(dims=[5, 5], init=lambda: 0.01)
bias2 = random()
weights3 = matrix.Matrix(dims=[10, 16], init=lambda: 0.01)
bias3 = matrix.Matrix(dims=[10, 1], init=lambda: 0.01)
weights4 = matrix.Matrix(dims=[10, 10], init=lambda: 0.01)
bias4 = matrix.Matrix(dims=[10, 1], init=lambda: 0.01)

l1 = convolutional.Convolutional(weights1, bias1, 2, 2, misc.relu)
l2 = convolutional.Convolutional(weights2, bias2, 2, 2, misc.relu)
h1 = fullyconnected.FullyConnected(weights3, bias3, misc.relu)
o = fullyconnected.FullyConnected(weights4, bias4, misc.relu)

for _ in range(2):
    for data in data_set[:10]:
        predict1 = l1.predict(data[0])
        predict2 = l2.predict(predict1)
        pred2Size = predict2.size()
        flattened = predict2.flatten().transpose()
        predict3 = h1.predict(flattened)
        output = o.predict(predict3)

        h_Errors1 = misc.getDifferences(misc.crossEntropy, output, data[1])
        h_Errors2 = h1.train(flattened, predict3, h_Errors1, misc.adam)
        h_Errors3 = l2.train(predict1, predict2, h_Errors2, misc.adam)
        h_ErrorUseless = l1.train(data[0], predict1, h_Errors3, misc.adam)

predict1 = l1.predict(data[0])
predict2 = l2.predict(predict1)
pred2Size = predict2.size()
flattened = predict2.flatten().transpose()
predict3 = h1.predict(flattened)
output = o.predict(predict3)
output.print()
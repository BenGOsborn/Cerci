import fullyconnected
import convolutional
import misc
import matrix
import mnist_parser

from random import random

data_set = mnist_parser.loadData("python/framework/data.pickle")

weights1 = matrix.Matrix(dims=[5, 5], init=lambda: random())
bias1 = random()
weights2 = matrix.Matrix(dims=[5, 5], init=lambda: random())
bias2 = random()
weights3 = matrix.Matrix(dims=[10, 16], init=lambda: random())
bias3 = matrix.Matrix(dims=[10, 1], init=lambda: random())
weights4 = matrix.Matrix(dims=[10, 10], init=lambda: random())
bias4 = matrix.Matrix(dims=[10, 1], init=lambda: random())

l1 = convolutional.Convolutional(weights1, bias1, 2, 2, misc.relu)
l2 = convolutional.Convolutional(weights2, bias2, 2, 2, misc.relu)
h1 = fullyconnected.FullyConnected(weights3, bias3, misc.relu)
o = fullyconnected.FullyConnected(weights4, bias4, misc.softmax)

for _ in range(1):
    for data in data_set[:1]:
        print("\n\nSTART")

        predict1 = l1.predict(data[0])
        print("PREDICTION 1")
        predict1.print()
        predict2 = l2.predict(predict1)
        print("PREDICTION 2")
        pred2Size = predict2.size()
        predict2.print()
        flattened = predict2.flatten().transpose()
        predict3 = h1.predict(flattened)
        print("PREDICTION 3")
        predict3.print()
        output = o.predict(predict3)
        print("PREDICT OUT")
        output.print()

        h_Errors1 = misc.getDifferences(misc.crossEntropy, output, data[1])
        print("ERRORS FIRST")
        h_Errors1.print()
        h_Errors2 = h1.train(flattened, predict3, h_Errors1, misc.adam)
        print("\nERRORS SECOND")
        h_Errors2.print()
        h_Errors3 = l2.train(predict1, predict2, h_Errors2, misc.adam)
        print("\nERRORS THIRD")
        h_Errors3.print()
        h_ErrorUseless = l1.train(data[0], predict1, h_Errors3, misc.adam)

# Must be some sort of problem with the way it takes the previos errors

predict1 = l1.predict(data_set[0][0])
predict2 = l2.predict(predict1)
pred2Size = predict2.size()
flattened = predict2.flatten().transpose()
predict3 = h1.predict(flattened)
output = o.predict(predict3)
print("OUT VALS")
output.print()
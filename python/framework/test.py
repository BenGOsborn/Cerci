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

# I need some way of parsing through the flattened layer and getting the hidden errrors from the first output layer for the kernels

l1 = convolutional.Convolutional(weights1, bias1, 1, 1, misc.relu)
l2 = convolutional.Convolutional(weights2, bias2, 1, 1, misc.relu)
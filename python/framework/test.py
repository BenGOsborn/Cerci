import fullyconnected
import convolutional
import misc
import matrix
import tensor
import mnist_parser

# data_set = mnist_parser.loadData("python/framework/data.pickle")

# weights1 = matrix.Matrix(dims=[2, 2], init=lambda: misc.weightRandom())
# bias1 = misc.weightRandom() 
# weights2 = matrix.Matrix(dims=[1, 6], init=lambda: misc.weightRandom())
# bias2 = matrix.Matrix(arr=[misc.weightRandom()])

# inputs = [matrix.Matrix(arr=[[0, 1, 0, 0],
#                              [0, 1, 0, 0],
#                              [0, 1, 0, 0]]),

#          matrix.Matrix(arr=[[0, 1, 1, 0],
#                             [0, 1, 1, 1],
#                             [0, 0, 0, 0]]),

#         matrix.Matrix(arr=[[1, 0, 0, 0],
#                            [1, 0, 0, 0],
#                            [1, 0, 0, 0]])
#         ]

# training = [matrix.Matrix(arr=[1]), matrix.Matrix(arr=[0]), matrix.Matrix(arr=[1])]

# c1 = convolutional.Conv2d(weights1, bias1, 1, 1, misc.relu)
# f1 = fullyconnected.FullyConnected(weights2, bias2, misc.sigmoid)

# # Errors could be happening with the transposing and the flattenng of the output layers being fed into the convolutional layer
# for _ in range(40):
#     for i in range(3):
#         cPredict = c1.predict(inputs[i])
#         flatC = cPredict.flatten().transpose()
#         predictF = f1.predict(flatC)

#         predictFErr = misc.getDifferences(misc.crossEntropy, predictF, training[i])
#         errorsC1 = f1.train(flatC, predictF, predictFErr, misc.adam)
#         c1.train(inputs[i], errorsC1, misc.adam, predicted=cPredict)

# cPredict = c1.predict(inputs[1])
# flatC = cPredict.flatten().transpose()
# predictF = f1.predict(flatC)
# print("OUTPUTS")
# predictF.print()

# print("FC NETWORK")
# f1.returnNetwork()[0].print()
# f1.returnNetwork()[3].print()
# print("C NETWORK")
# c1.returnNetwork()[0].print()
# print([c1.returnNetwork()[3]])

weightsFilters = tensor.ConvFilter(4, 4, 1, 3)
biasTensor = tensor.BiasConvTensor(1, 3)

block = convolutional.Conv(weightsFilters, biasTensor, 1, 1, misc.relu)
flatten = convolutional.Flatten()

mat = matrix.Matrix(dims=[7, 7], init=lambda: 0.5)
inputs = tensor.Tensor([mat])

pred = block.predict(inputs)
flattened = flatten.flatten(pred)

# How am I going to create  flattening layer for this? It will probably relate somewhat back to the pooling layer too?
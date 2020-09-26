import fullyconnected
import convolutional
import misc
import matrix
import tensor
import mnist_parser

data_set = mnist_parser.loadData("python/framework/data.pickle")

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

weightsFilters = tensor.ConvFilter(5, 5, 1, 1)
biasTensor = tensor.BiasConvTensor(1, 1)
weight_set = matrix.Matrix(dims=[10, 16], init=lambda: misc.weightRandom())
bias_set = matrix.Matrix(dims=[10, 1], init=lambda: misc.weightRandom())

# Check the dropout rates for everything
# Do a cleanup of the codebase and add more regularization and normalization functions
# Need to put dropout after the activation function
block = convolutional.Conv(weightsFilters, biasTensor, 1, 1, misc.relu, dropout_rate=0.1)
poolLayer = convolutional.Pool(5, 5, 5, 5)
flatten = convolutional.Flatten()
fc = fullyconnected.FullyConnected(weight_set, bias_set, misc.softmax)

inp = data_set[0][0]
inputs = tensor.Tensor([inp])
training = data_set[0][1]

for _ in range(20):
    for dt in data_set[:10]:
        inputs = tensor.Tensor([dt[0]])
        training = dt[1]

        pred = block.predict(inputs)
        pooled = poolLayer.pool(pred)
        flattened = flatten.flatten(pooled).transpose()
        predOut = fc.predict(flattened)

        predOutErr = misc.getDifferences(misc.crossEntropy, predOut, training)
        hiddenFlat = fc.train(flattened, predOut, predOutErr, misc.adam)
        pooledError = flatten.reshapeErrors(hiddenFlat.transpose())
        unPooled = poolLayer.reshapeErrors(pooledError)
        block.train(inputs, pred, unPooled, misc.adam)

predOut.transpose().print()
training.print()

# The network is slow how am I going to be able to fix this?
# I should include in the tensor class and the matrix class the ability to look at values in the matrix without having to return all of the time even though its in multiple dimensions
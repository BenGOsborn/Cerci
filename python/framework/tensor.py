import matrix
from random import random

# This only deals with tensors of length 3
def tensorSum(tensor):
    tensors = tensor.returnTensor()
    sumMatrix = tensors[0]
    for tser in tensors[1:]:
        sumTotal = matrix.add(sumMatrix, tser)

    return sumTotal

class Tensor:
    def validTensor(self):
        try:
            self.__tensor[0]
        except:
            self.__tensor = [self.__tensor]
        mSize = self.__tensor[0].size()
        for mat in self.__tensor:
            if (mat.size() != mSize): raise Exception("Matrices must all be of same size!")

    def __init__(self, tensor):
        self.__tensor = tensor 
        self.validTensor()

    def print(self):
        for mat in self.__tensor:
            mat.print()
            print()

    def clone(self):
        return Tensor(self.__tensor)

    def returnTensor(self):
        return self.__tensor

    def size(self):
        matSize = self.__tensor[0].size()
        return matSize[0], matSize[1], len(self.__tensor)

class ConvFilter:
    def __init__(self, row_size, col_size, filter_depth, filter_number):
        self.__filters = []
        for _ in range(filter_number):
            genMatrix = [matrix.Matrix(dims=[row_size, col_size], init=lambda: random()-0.5) for _ in range(filter_depth)]
            matrixTensor = Tensor(genMatrix)
            self.__filters.append(matrixTensor)

        self.__row_size = row_size
        self.__col_size = col_size
        self.__filter_depth = filter_depth
        self.__filter_number = filter_number

    def print(self):
        for tser in self.__filters:
            tser.print()
            print()

    def returnFilters(self):
        return self.__filters

    def size(self):
        return self.__row_size, self.__col_size, self.__filter_depth, self.__filter_number

# Is there a better way to do this through inheritance instead?
class BiasConvTensor:
    def __init__(self, filter_depth, filter_number):
        self.__biasTensor = []
        for _ in range(filter_number):
            biasMatrix = matrix.Matrix(dims=[1, filter_depth], init=lambda: random()-0.5)
            self.__biasTensor.append(biasMatrix)

        self.__filter_depth = filter_depth
        self.__filter_number = filter_number

    def print(self):
        for mat in self.__biasTensor:
            mat.print()
            print()

    def returnTensor(self):
        return self.__biasTensor

    def returnBias(self):
        return self.__filter_depth, self.__filter_number
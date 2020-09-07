# I want to create a dot product library
# I want to create a sigmoid function and a sigmoid derivative
from math import exp
from random import random

def dot(arr1, arr2):
    if (len(arr1) != len(arr2)): raise Exception(f"Arrays are not of same length! Arr1 Length: {len(arr1)} | Arr2 Length: {len(arr2)}")
    return sum([one*two for one, two in zip(arr1, arr2)])

def error(arr1, arr2):
    if (len(arr1) != len(arr2)): raise Exception(f"Arrays are not of same length! Arr1 Length: {len(arr1)} | Arr2 Length: {len(arr2)}")
    return abs(sum([one-two for one, two in zip(arr1, arr2)]))

def sigmoid(x, deriv=False):
    if not deriv:
        return 1/(1+exp(-x))
        # This is because we pass through the sigmoid value, and therefore the value is not raw and does not need to be parsed as a sigmoid value as it already is one
    return x*(1-x)

# One thing to remember is that the values can have multiple outputs but they just have to be put into array values

# For a single layer network

def trainData():
    # For a single layer networ
    weightsSingle = [
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    ]
    biasSingle = [
        [0.5, 0.5, 0.5]
    ]

    # For a double layer network
    weightsDouble = [
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        [[0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.4, 0.4, 0.4]]
    ]
    biasDouble = [
        [0.5, 0.5, 0.5],
        [0.4, 0.4, 0.4]
    ]

    # For a multi layer network
    weightsMulti = [
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    ]
    biasMulti = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ]

    return {
            "weightsSingle": weightsSingle, "biasSingle": biasSingle, 
            "weightsDouble": weightsDouble, "biasDouble": biasDouble, 
            "weightsMulti": weightsMulti, "biasMulti": biasMulti
            }
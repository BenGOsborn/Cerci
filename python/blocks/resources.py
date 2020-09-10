from random import random
from math import exp, tanh

def dot(arr1, arr2):
    if (len(arr1) != len(arr2)): raise Exception(f"Arrays are not of same length! Arr1 Length: {len(arr1)} | Arr2 Length: {len(arr2)}")
    return sum([one*two for one, two in zip(arr1, arr2)])

def error(arr1, arr2):
    if (len(arr1) != len(arr2)): raise Exception(f"Arrays are not of same length! Arr1 Length: {len(arr1)} | Arr2 Length: {len(arr2)}")
    return sum([abs(one - two) for one, two in zip(arr1, arr2)])

def relu(x, deriv=False):
    if not deriv:
        return x if x > 0 else 0.1*x
        # This is because we pass through the eigmoid value, and therefore the value is not raw and does not need to be parsed as a eigmoid value as it already is one
    return 1 if x > 0 else 0.1

def sigmoid(x, deriv=False):
    if not deriv:
        return 1/(1+exp(-x))
    return x*(1-x)

def learnFunc(x):
    return 0.5*abs(tanh(4*x))

# One thing to remember is that the values can have multiple outputs but they just have to be put into array values

# For a single layer network

def trainData():
    # For a single layer networ
    weightsSingle = [
        [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]]
    ]
    biasSingle = [
        random()
    ]

    # For a double layer network
    weightsDouble = [
        [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()], [random(), random(), random()], [random(), random(), random()]],
        [[random(), random(), random(), random(), random()], [random(), random(), random(), random(), random()], [random(), random(), random(), random(), random()]]
    ]
    biasDouble = [
        random(),
        random()
    ]

    # For a multi layer network
    weightsMulti = [
        [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]],
        [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]],
        [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]]
    ]
    biasMulti = [
        random(),
        random(),
        random()
    ]

    return {
            "weightsSingle": weightsSingle, "biasSingle": biasSingle, 
            "weightsDouble": weightsDouble, "biasDouble": biasDouble, 
            "weightsMulti": weightsMulti, "biasMulti": biasMulti
            }
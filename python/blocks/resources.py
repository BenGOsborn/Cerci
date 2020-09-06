# I want to create a dot product library
# I want to create a sigmoid function and a sigmoid derivative
from math import exp

def dot(arr1, arr2):
    if (len(arr1) != len(arr2)): raise Exception(f"Arrays are not of same length! Arr1 Length: {len(arr1)} | Arr2 Length: {len(arr2)}")
    return sum([one*two for one, two in zip(arr1, arr2)])

def sigmoid(x, deriv=False):
    if not deriv:
        return 1/(1+exp(-x))
        # This is because we pass through the sigmoid value, and therefore the value is not raw and does not need to be parsed as a sigmoid value as it already is one
    return x*(1-x)
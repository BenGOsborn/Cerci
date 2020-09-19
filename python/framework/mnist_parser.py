import pickle
import matrix
import numpy as np
from random import shuffle

def parser(val, num_options=10):
        empty_row = [0 for _ in range(num_options)]
        empty_row[int(val-1)] = 1

        return empty_row

def genData(raw_data_file_path, data_size):
    data_raw = np.loadtxt(raw_data_file_path, delimiter=",", max_rows=data_size)
    
    data_set = []
    for i in range(data_size):
        data = matrix.Matrix(arr=list(data_raw[i][1:].reshape(28, 28)))
        label = matrix.Matrix(arr=parser(data_raw[i][0]))

        data_set.append((data, label))

    with open("data.pickle", "wb") as f:
        pickle.dump(data_set, f)

    return data_set

def loadData(data_path, shuffleData=False):
    with open(data_path, "rb") as f:
        data_set = pickle.load(f)

    if (shuffleData):
        shuffle(data_set)
    return data_set

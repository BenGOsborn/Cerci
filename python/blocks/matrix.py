# Takes the dot product of two input vectors
def dot(vector1, vector2):
    if (len(vector1) != len(vector2)): raise Exception(f"Vectors are not of same length! Length vector 1: {len(vector1)} | Length vector 2: {len(vector2)}") 
    return sum([val1*val1 for val1, val2 in zip(vector1, vector2)])

# Add 1 to 2
def add(matrix1, matrix2):
    if (matrix1.size() != matrix2.size()): raise Exception(f"Matrices must be same size! Matrix size 1: {matrix1.size()} | Matrix size 2: {matrix2.size()}")

    mat1 = matrix1.returnMatrix()
    mat2 = matrix2.returnMatrix()

    matrixNew = []
    for y in range(matrix1.size()[1]):
        tempArr = []
        for x in range(matrix1.size()[0]):
            val = mat2[y][x] + mat1[y][x]
            tempArr.append(val)
        matrixNew.append(tempArr)

    return Matrix(arr=matrixNew)

# Subtract 1 from 2
def subtract(matrix1, matrix2):
    if (matrix1.size() != matrix2.size()): raise Exception(f"Matrices must be same size! Matrix size 1: {matrix1.size()} | Matrix size 2: {matrix2.size()}")

    mat1 = matrix1.returnMatrix()
    mat2 = matrix2.returnMatrix()

    matrixNew = []
    for y in range(matrix1.size()[1]):
        tempArr = []
        for x in range(matrix1.size()[0]):
            val = mat2[y][x] - mat1[y][x]
            tempArr.append(val)
        matrixNew.append(tempArr)

    return Matrix(arr=matrixNew)

# Returns the scalar multiplication of a matrix and a factor
def multiplyScalar(matrix, factor):
    newMatrix = []
    for y in range(matrix.size()[0]):
        tempArr = []
        for x in range(matrix.size()[1]):
            val = factor*matrix[y][x] 
            tempArr.append(val)
        newMatrix.append(tempArr)

    return Matrix(arr=newMatrix)

# Returns matrix1*matrix2
def multiplyMatrices(matrix1, matrix2):
    if (matrix1.size() != matrix2.size()): raise Exception(f"Matrix 1's columns must be equal length to Matrix 2's Rows! Matrix 1's columns: {matrix1.size()[1]} | Matrix 2's rows: {matrix2.size()[0]}")

    mat1 = matrix1.returnMatrix()
    mat2 = matrix2.returnMatrix()

    matrixNew = []
    for y in range(matrix1.size()[0]):
        tempArr = []
        for x in range(matrix2.size()[1]):
            splicedMat2 = [mat2[i][x] for i in range(matrix2.size()[0])]
            val = dot(mat1[y], splicedMat2)
            tempArr.append(val)
        matrixNew.append(tempArr)

    return Matrix(arr=matrixNew)

# For optimization preallocate the length of the matrices and then add in the values by index
class Matrix:
    def __init__(self, arr=False, dims=False):
        if ((dims == False) and (arr == False)):
            raise Exception("Requires a n*n array or an array containing an integer as the rows count and the columns count!")
        elif ((dims != False) and (arr != False)):
            raise Exception("Matrix can only accept one input style!")
        elif (arr != False):
            self.__matrix = arr
            try:
                self.__matrix[0][0][0]
                raise Exception("Matrices must be of dimension n*n")
            except:
                try:
                    self.__matrix[0][0]
                except:
                    raise Exception("Matrices must be of dimension n*n")
        else:
            if (len(dims) != 2):
                raise Exception("'dims' requires an array of 2 parameters: the rows count and the columns count")
            self.__matrix = [[1 for _ in range(dims[1])] for _ in range(dims[0])]

    def print(self):
        for row in self.__matrix:
            print(row)

    def transpose(self):
        new_matrix = [[0 for _ in range(len(self.__matrix))] for _ in range(len(self.__matrix[0]))]

        for y in range(len(self.__matrix)):
            for x in range(len(self.__matrix[0])):
                new_matrix[x][y] = self.__matrix[y][x]

        self.__matrix = new_matrix

    def applyFunc(self, func):
        for y in range(len(self.__matrix)):
            for x in range(len(self.__matrix[0])):
               self.__matrix[y][x] = func(self.__matrix[y][x])

    def returnMatrix(self):
        return self.__matrix

    def size(self):
        return [len(self.__matrix), len(self.__matrix[0])]
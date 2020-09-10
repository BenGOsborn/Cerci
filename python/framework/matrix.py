# Takes the dot product of two input vectors
def dot(vector1, vector2):
    if (len(vector1) != len(vector2)): raise Exception(f"Vectors are not of same length! Length vector 1: {len(vector1)} | Length vector 2: {len(vector2)}") 
    return sum([val1*val2 for val1, val2 in zip(vector1, vector2)])

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
            val = mat1[y][x] - mat2[y][x]
            tempArr.append(val)
        matrixNew.append(tempArr)

    return Matrix(arr=matrixNew)

# Returns the scalar multiplication of a matrix and a factor
def multiplyScalar(matrix, factor):
    mat = matrix.returnMatrix()

    newMatrix = []
    for y in range(matrix.size()[0]):
        tempArr = []
        for x in range(matrix.size()[1]):
            val = factor*mat[y][x] 
            tempArr.append(val)
        newMatrix.append(tempArr)

    return Matrix(arr=newMatrix)

# Returns matrix1*matrix2
def multiplyMatrices(matrix1, matrix2):
    if (matrix1.size()[1] != matrix2.size()[0]): raise Exception(f"Matrix 1's columns must be equal length to Matrix 2's Rows! Matrix 1's columns: {matrix1.size()[1]} | Matrix 2's rows: {matrix2.size()[0]}")

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
    def validMatrix(self):
        try:
            self.__matrix[0][0]
        except:
            self.__matrix = [self.__matrix]

    def __init__(self, arr=False, dims=False):
        if (arr != False):
            self.__matrix = arr
            self.validMatrix()
        elif (dims != False):
            self.__matrix = [[0.5 for _ in range(dims[1])] for _ in range(dims[0])] # Rows and columns (Rows is the height, colums is the row length)
            self.validMatrix()
        else:
            raise Exception("Matrix requires parameter 'arr' or 'dims'!")

    def print(self):
        for row in self.__matrix:
            print(row)
            
    def flatten(self):
        new_matrix = []
        
        for row in self.__matrix:
            for val in row:
                new_matrix.append(val)
                
        self.__matrix = new_matrix
        self.validMatrix()
            
    def transpose(self):
        new_matrix = [[0 for _ in range(len(self.__matrix))] for _ in range(len(self.__matrix[0]))]

        for y in range(len(self.__matrix)):
            for x in range(len(self.__matrix[0])):
                new_matrix[x][y] = self.__matrix[y][x]

        self.__matrix = new_matrix
        self.validMatrix()

    def average(self):
        new_matrix = Matrix(arr=self.__matrix)
        new_matrix.flatten()
        ln = new_matrix.size()[1]
        sm = sum(new_matrix.returnMatrix()[0])
        avg = sm/ln

        return avg

    def applyFunc(self, func):
        for y in range(len(self.__matrix)):
            for x in range(len(self.__matrix[0])):
               self.__matrix[y][x] = func(self.__matrix[y][x])

        self.validMatrix()

    def returnMatrix(self):
        return self.__matrix

    def size(self):
        return [len(self.__matrix), len(self.__matrix[0])]
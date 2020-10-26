class Tensor:
    def __init__(self, tensor, shape):
        self.dims = len(shape)

        product = 1
        for i in range(self.dims):
            product *= shape[i]
        assert(product == len(tensor))

        self.tensor = tensor
        self.shape = shape

    def __str__(self):
        return self.__string()

    def __string(self, index=-1, position=0):
        # This is the base case that will occur if it is the last node in the tree
        if (abs(index) == self.dims):
            mat = "[ "
            for i in range(self.shape[0]):
                mat += f"{self.tensor[position + i]} "
            mat += "]"
            return mat
        
        mat_final = "[ "

        product = 1
        for i in range(self.dims + index):
            product *= self.shape[i]

        for i in range(self.shape[index]):
            mat_final += f"\n{abs(index) * ' '}{ self.__string(index-1, position+product*i)} "
        
        return f"{mat_final}\n{(abs(index) - 1) * ' '}]" if (index != -1) else f"{mat_final}\n]"

    def reshape(self, new_shape):
        assert(self.dims == len(new_shape))

        product = 1
        for i in range(len(new_shape)):
            product *= new_shape[i]
        assert(product == len(self.tensor))

        return Tensor(self.tensor.copy(), self.shape.copy())

    def returnTensor(self):
        return self.tensor.copy()

t = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 2, 2, 2])
print(t)
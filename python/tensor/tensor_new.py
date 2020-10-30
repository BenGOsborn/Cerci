class AddElementwise:

    @staticmethod
    def forward(matrix_left, matrix_right, backwards=False):
        assert(matrix_left.shape == matrix_right.shape)
        new_tensor = [a+b for a, b in zip(matrix_left.tensor, matrix_right.tensor)]

        if (not backwards):
            return Tensor(new_tensor, matrix_left.shape, left=matrix_left, right=matrix_right, 
                            track_grad=(matrix_left.track_grad or matrix_right.track_grad), operator=AddElementwise)

        return Tensor(new_tensor, matrix_left.shape, left=None, right=None, 
                        track_grad=False, operator=None)

    @staticmethod
    def ddleft(matrix_left, matrix_right):
        assert(matrix_left.shape == matrix_right.shape)
        new_tensor = [1 for _ in range(matrix_left.size)] 

        return Tensor(new_tensor, matrix_left.shape, left=None, right=None,
                       track_grad=False, operator=None)

    @staticmethod
    def ddright(matrix_left, matrix_right):
        assert(matrix_left.shape == matrix_right.shape)
        new_tensor = [1 for _ in range(matrix_left.size)] 

        return Tensor(new_tensor, matrix_left.shape, left=None, right=None,
                       track_grad=False, operator=None)

class MultiplyElementwise:

    @staticmethod
    def forward(matrix_left, matrix_right, backwards=False):
        assert(matrix_left.shape == matrix_right.shape)
        new_tensor = [a*b for a, b in zip(matrix_left.tensor, matrix_right.tensor)]

        if (not backwards):
            return Tensor(new_tensor, matrix_left.shape, left=matrix_left, right=matrix_right, 
                            track_grad=(matrix_left.track_grad or matrix_right.track_grad), operator=MultiplyElementwise)
        
        return Tensor(new_tensor, matrix_left.shape, left=None, right=None, 
                        track_grad=False, operator=None)

    @staticmethod
    def ddleft(matrix_left, matrix_right):
        assert(matrix_left.shape == matrix_right.shape)

        return Tensor(matrix_left.tensor.copy(), matrix_left.shape.copy(), left=None, right=None, 
                       track_grad=False, operator=None)

    @staticmethod
    def ddright(matrix_left, matrix_right):
        assert(matrix_left.shape == matrix_right.shape)

        return Tensor(matrix_right.tensor.copy(), matrix_right.shape.copy(), left=None, right=None,
                       track_grad=False, operator=None)

class TensorBase:
    def __init__(self, tensor, shape): # The left and the right will contain the links to the other nodes in the tree
        self.dims = len(shape)
        self.size = len(tensor)

        check_length = 1
        for i in range(self.dims):
            check_length *= shape[i]
        assert(check_length == self.size)

        self.tensor = tensor
        self.shape = shape
    
    def __str__(self):
        return self.__string()

    # There is probably a more efficient way to do this
    def __string(self, index=-1, position=0):
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

    def __mul__(self, other):
        return MultiplyElementwise.forward(self, other)

class Tensor(TensorBase):
    def __init__(self, tensor, shape, left=None, right=None, track_grad=False, operator=None):
        super().__init__(tensor, shape)

        self.track_grad = track_grad

        if (track_grad):
            self.operator = operator

            self.left = left
            self.right = right
            self.grad = Tensor([0 for _ in range(self.size)], self.shape)

    def zeroGrad(self):
        if (self.track_grad):
            self.grad = Tensor([0 for _ in range(self.size)], self.shape)
            if (self.left != None):
                self.left.zeroGrad()
            if (self.right != None):
                self.right.zeroGrad()

    # Now I need to do the actual backwards function
    def backwards(self, factors=None):
        if (factors == None):
            factors = Tensor([1 for _ in range(self.size)], self.shape)
        self.grad = AddElementwise.forward(self.grad, factors, backwards=True)

        # I cant just use none when I port this to C++
        if (self.left != None):
            self.left.backwards(factors=MultiplyElementwise.forward(factors, self.operator.ddleft(self.left, self.right)))
        if (self.right != None):
            self.right.backwards(factors=MultiplyElementwise.forward(factors, self.operator.ddright(self.left, self.right)))
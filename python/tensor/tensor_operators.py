import tensor

class AddElementwise:

    @staticmethod
    def forward(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        new_tensor = [a+b for a, b in zip(matrix_a.tensor, matrix_b.tensor)]

        return tensor.Tensor(new_tensor, matrix_a.shape)

    @staticmethod
    def dda(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        ret_tensor = [1 for _ in range(matrix_a.size)]

        return tensor.Tensor(ret_tensor, matrix_a.shape)

    @staticmethod
    def ddb(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        ret_tensor = [1 for _ in range(matrix_a.size)]

        return tensor.Tensor(ret_tensor, matrix_a.shape)

class MultiplyElementwise:

    @staticmethod
    def forward(matrix_a, matrix_b):
        assert(matrix_a.shape == matrix_b.shape)
        new_tensor = [a*b for a, b in zip(matrix_a.tensor, matrix_b.tensor)]

        return tensor.Tensor(new_tensor, matrix_a.shape)

    @staticmethod
    def dda(matrix_a, matrix_b):
        return matrix_b

    @staticmethod
    def ddb(matrix_a, matrix_b):
        return matrix_a
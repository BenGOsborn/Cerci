#include "matrix.cuh"

Matrix::Matrix(std::unique_ptr<float[]>& inMatrix, std::unique_ptr<int[]>& inShape) {
	Matrix::shape = std::make_unique<int[]>(2);
	memcpy(shape.get(), inShape.get(), 2 * sizeof(int));

	Matrix::size = std::make_unique<int>(shape[0] * shape[1]);

	Matrix::matrix = std::make_unique<float[]>(*size);
	memcpy(matrix.get(), inMatrix.get(), *size * sizeof(float));
}

void Matrix::print() {
	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			std::cout << matrix[i * shape[1] + j] << " ";
		}
		std::cout << "\n";
	}
}

std::unique_ptr<Matrix> Matrix::reshape(int rows, int cols) {
	if (rows * cols != *size) throw std::invalid_argument("New matrix size does not match original matrix size!");

	std::unique_ptr<int[]> new_shape = std::make_unique<int[]>(2);
	new_shape[0] = rows;
	new_shape[1] = cols;

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(matrix, new_shape);

	return ret_matrix;
}

__global__
void transposeD(int rows, int cols, float* inVector) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < rows) && (col < cols)) inVector[col * rows + row] = inVector[row * cols + col];
}

std::unique_ptr<Matrix> Matrix::transpose() {
	std::unique_ptr<int[]> new_shape = std::make_unique<int[]>(2);
	new_shape[0] = shape[1];
	new_shape[1] = shape[0];

	int bytes = *size * sizeof(float);

	float* dCopy;
	cudaMalloc(&dCopy, bytes);
	cudaMemcpy(dCopy, matrix.get(), bytes, cudaMemcpyHostToDevice);

	int grid_rows = (shape[0] + Matrix::gpu->BLOCK_SIZE - 1) / Matrix::gpu->BLOCK_SIZE;
	int grid_cols = (shape[1] + Matrix::gpu->BLOCK_SIZE - 1) / Matrix::gpu->BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(Matrix::gpu->BLOCK_SIZE, Matrix::gpu->BLOCK_SIZE);

	transposeD <<< dimGrid, dimBlock >>> (shape[0], shape[1], dCopy);

	std::unique_ptr<float[]> new_matrix = std::make_unique<float[]>(*size);
	cudaMemcpy(new_matrix.get(), dCopy, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(new_matrix, new_shape);

	cudaFree(dCopy);

	return ret_matrix;
}

std::unique_ptr<Matrix> Matrix::clone() {
	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(matrix, shape);
	return ret_matrix;
}

std::unique_ptr<float[]> Matrix::returnMatrix() {
	std::unique_ptr<float[]> ret_matrix = std::make_unique<float[]>(*size);
	memcpy(ret_matrix.get(), matrix.get(), *size * sizeof(float));

	return ret_matrix;
}

std::unique_ptr<int[]> Matrix::returnShape() {
	std::unique_ptr<int[]> ret_shape = std::make_unique<int[]>(2);
	memcpy(ret_shape.get(), shape.get(), 2 * sizeof(unsigned int));
	return ret_shape;
}

int Matrix::returnSize() {
	return *size;
}

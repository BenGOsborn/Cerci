#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>

int NUM_THREADS = 1 << 10;

template <typename Lambda>
__global__ 
void applyD(int size, float* inVector, Lambda function) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) inVector[idx] = function(inVector[idx]);
}

__global__
void transposeD(int rows, int cols, float* inVector) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if ((idy < rows) && (idx < cols)) inVector[idx * cols + idy] = inVector[idy * rows + idx];
}

class Matrix {
private:
	std::unique_ptr<float[]> matrix;
	std::unique_ptr<int> size;
	std::unique_ptr<int[]> shape;

	// If I remove the macro then I should redefine the 'NUM_THREADS'
	std::unique_ptr<int> NUM_BLOCKS;

public:
	Matrix(std::unique_ptr<float[]>& inMatrix, std::unique_ptr<int[]>& inShape) {
		shape = std::make_unique<int[]>(2);
		memcpy(shape.get(), inShape.get(), 2 * sizeof(int));

		size = std::make_unique<int>(shape[0] * shape[1]);

		matrix = std::make_unique<float[]>(*size);
		memcpy(matrix.get(), inMatrix.get(), *size * sizeof(float));

		// This constant 'NUM_THREADS' can be adjusted
		NUM_BLOCKS = std::make_unique<int>((*size + NUM_THREADS - 1) / NUM_THREADS);
	}

	void print() {
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				std::cout << matrix[i * shape[1] + j] << " ";
			}
			std::cout << "\n";
		}
	}

	std::unique_ptr<Matrix> reshape(int rows, int cols) {
		if (rows * cols != *size) throw std::invalid_argument("New matrix size does not match original matrix size!");

		std::unique_ptr<int[]> new_shape = std::make_unique<int[]>(2);
		new_shape[0] = rows;
		new_shape[1] = cols;

		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(matrix, new_shape);

		return ret_matrix;
	}

	// This could be done in parallel on the GPU
	std::unique_ptr<Matrix> transpose() {
		std::unique_ptr<int[]> new_shape = std::make_unique<int[]>(2);
		new_shape[0] = shape[1];
		new_shape[1] = shape[0];

		int bytes = *size * sizeof(float);

		float* dCopy;
		cudaMalloc(&dCopy, bytes);
		cudaMemcpy(dCopy, matrix.get(), bytes, cudaMemcpyHostToDevice);

		transposeD <<< *NUM_BLOCKS, NUM_THREADS >>> (shape[0], shape[1], dCopy);

		std::unique_ptr<float[]> new_matrix = std::make_unique<float[]>(*size);
		cudaMemcpy(new_matrix.get(), dCopy, bytes, cudaMemcpyDeviceToHost);

		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(new_matrix, new_shape);

		cudaFree(dCopy);

		return ret_matrix;
	}

	std::unique_ptr<Matrix> clone() {
		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(matrix, shape);
		return ret_matrix;
	}

	// I cant run this on my GPU yet for some reason so it has to be done on the CPU I will get it working though
	template <typename Lambda>
	std::unique_ptr<Matrix> apply(Lambda function) {
		int bytes = *size * sizeof(float);

		float* dCopy;
		cudaMalloc(&dCopy, bytes);
		cudaMemcpy(dCopy, matrix.get(), bytes, cudaMemcpyHostToDevice);

		applyD <<< *NUM_BLOCKS, NUM_THREADS >>> (*size, dCopy, function);

		std::unique_ptr<float[]> new_matrix = std::make_unique<float[]>(*size);
		cudaMemcpy(new_matrix.get(), dCopy, bytes, cudaMemcpyDeviceToHost);

		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(new_matrix, shape);

		cudaFree(dCopy);

		return ret_matrix;
	}

	std::unique_ptr<float[]> returnMatrix() {
		std::unique_ptr<float[]> ret_matrix = std::make_unique<float[]>(*size);
		memcpy(ret_matrix.get(), matrix.get(), *size * sizeof(float));

		return ret_matrix;
	}

	std::unique_ptr<int[]> returnShape() {
		std::unique_ptr<int[]> ret_shape = std::make_unique<int[]>(2);
		memcpy(ret_shape.get(), shape.get(), 2 * sizeof(unsigned int));
		return ret_shape;
	}

	int returnSize() {
		return *size;
	}
};

__global__
void addD(int size, float* vector1, float *vector2, float *retVector) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) retVector[idx] = vector1[idx] + vector2[idx];
}

// For subtraction just multiply the array by 1 the add into a negative which can be done through the apply func
std::unique_ptr<Matrix> add(std::unique_ptr<Matrix> &matrix1, std::unique_ptr<Matrix> &matrix2) {
	std::unique_ptr<int[]> mat1shape = matrix1->returnShape();
	std::unique_ptr<int[]> mat2shape = matrix2->returnShape();
	if ((mat1shape[0] != mat2shape[0]) || (mat1shape[1] != mat2shape[1])) throw std::invalid_argument("Matrices are not of the same shape!");

	int size = matrix1->returnSize();
	int bytes = size * sizeof(float);

	std::unique_ptr<float[]> mat1 = matrix1->returnMatrix();
	std::unique_ptr<float[]> mat2 = matrix2->returnMatrix();

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, bytes);
	cudaMalloc(&mat2d, bytes);
	cudaMalloc(&mat3d, bytes);
	cudaMemcpy(mat1d, mat1.get(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2.get(), bytes, cudaMemcpyHostToDevice);

	int NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;

	addD <<< NUM_BLOCKS, NUM_THREADS >>> (size, mat1d, mat2d, mat3d);

	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(bytes);
	cudaMemcpy(mat3.get(), mat3d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<int[]> shape = matrix1->returnShape();
	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, shape);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

__global__
// What are the specifications required for matrix multiplication...?
void multiplyD(int m, int n, int k, float* vector1, float* vector2, float* retVector) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if ((col < k) && (row < m)) {
		float sum = 0;
		for (int i = 0; i < n; i++) {
			sum += vector1[row * n + i] * vector2[i * k + col];
		}
		retVector[row * k + col] = sum;
	}
}

std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix> &matrix1, std::unique_ptr<Matrix> &matrix2) {
	std::unique_ptr<int[]> mat1shape = matrix1->returnShape();
	std::unique_ptr<int[]> mat2shape = matrix2->returnShape();
	if (mat1shape[1] != mat2shape[0]) throw std::invalid_argument("Matrix1's cols must equal Matrix2's rows!");

	std::unique_ptr<int[]> new_shape = std::make_unique<int[]>(2);
	new_shape[0] = mat1shape[0];
	new_shape[1] = mat2shape[1];
	int same = mat1shape[1];

	int new_size = new_shape[0] * new_shape[1];

	int mat1bytes = matrix1->returnSize() * sizeof(float);
	int mat2bytes = matrix2->returnSize() * sizeof(float);
	int mat3bytes = new_size * sizeof(float);

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, mat1bytes);
	cudaMalloc(&mat2d, mat2bytes);
	cudaMalloc(&mat3d, mat3bytes);

	std::unique_ptr<float[]> mat1 = matrix1->returnMatrix();
	std::unique_ptr<float[]> mat2 = matrix2->returnMatrix();
	cudaMemcpy(mat1d, mat1.get(), mat1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2.get(), mat2bytes, cudaMemcpyHostToDevice);

	int NUM_BLOCKS = (new_size + NUM_THREADS - 1) / NUM_THREADS;
	multiplyD <<< NUM_BLOCKS, NUM_THREADS >>> (new_shape[0], same, new_shape[1], mat1d, mat2d, mat3d);

	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(new_size);
	cudaMemcpy(mat3.get(), mat3d, mat3bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, new_shape);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

// Make some functions which can initialize matrices for us now

int main() {
	std::unique_ptr<int[]> shape1 = std::make_unique<int[]>(2);
	shape1[0] = 2;
	shape1[1] = 2;
	std::unique_ptr<float[]> vals1 = std::make_unique<float[]>(10);
	for (int i = 0; i < 10; i++) {
		vals1[i] = 2.0f;
	}
	std::unique_ptr<Matrix> matrix1 = std::make_unique<Matrix>(vals1, shape1);

	std::unique_ptr<int[]> shape2 = std::make_unique<int[]>(2);
	shape2[0] = 2;
	shape2[1] = 2;
	std::unique_ptr<float[]> vals2 = std::make_unique<float[]>(10);
	for (int i = 0; i < 10; i++) {
		vals2[i] = 3.0f;
	}
	std::unique_ptr<Matrix> matrix2 = std::make_unique<Matrix>(vals2, shape2);

	std::unique_ptr<Matrix> multiplied = multiply(matrix1, matrix2);
	multiplied->print();
}

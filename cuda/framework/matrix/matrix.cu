#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>

// This is meant to be the function that applies the functions on the gpu except it doesnt work
//__global__ template <typename F>
//void applyFuncD(int size, float* inVector, F func) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < size) inVector[idx] = func(inVector[idx]);
//}

class Matrix {
private:
	std::unique_ptr<float[]> matrix;
	std::unique_ptr<int> size;
	std::unique_ptr<int[]> shape;

public:
	Matrix(std::unique_ptr<float[]>& inMatrix, std::unique_ptr<int[]>& inShape) {
		shape = std::make_unique<int[]>(2);
		memcpy(shape.get(), inShape.get(), 2 * sizeof(int));

		size = std::make_unique<int>(shape[0] * shape[1]);

		matrix = std::make_unique<float[]>(*size);
		memcpy(matrix.get(), inMatrix.get(), *size * sizeof(float));
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

		std::unique_ptr<float[]> new_matrix = std::make_unique<float[]>(*size);
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				new_matrix[j * new_shape[1] + i] = matrix[i * shape[1] + j];
			}
		}

		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(new_matrix, new_shape);
		return ret_matrix;
	}

	std::unique_ptr<Matrix> clone() {
		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(matrix, shape);
		return ret_matrix;
	}

	// I cant run this on my GPU yet for some reason so it has to be done on the CPU I will get it working though
//	template <typename F>
//	std::unique_ptr<Matrix> applyFunc(F func) {
//		int bytes = *size * sizeof(float);
//
//		float* dCopy;
//		cudaMalloc(&dCopy, bytes);
//		cudaMemcpy(dCopy, matrix.get(), bytes, cudaMemcpyHostToDevice);
//
//		int NUM_THREADS = 1 << 10;
//		int NUM_BLOCKS = (*size + NUM_THREADS - 1) / NUM_THREADS;
//
//		applyFuncD <<< (NUM_BLOCKS, NUM_THREADS) >>> (*size, dCopy, func);
//
//		std::unique_ptr<float[]> new_matrix = std::make_unique<float[]>(*size);
//		cudaMemcpy(new_matrix.get(), dCopy, bytes, cudaMemcpyDeviceToHost);
//
//		std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(new_matrix, shape);
//
//		cudaFree(dCopy);
//
//		return ret_matrix;
//	}

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
std::unique_ptr<Matrix> addMatrices(std::unique_ptr<Matrix> &matrix1, std::unique_ptr<Matrix> &matrix2) {
	std::unique_ptr<int[]> mat1shape = matrix1->returnShape();
	std::unique_ptr<int[]> mat2shape = matrix2->returnShape();
	if ((mat1shape[0] != mat2shape[0]) || (mat1shape[1] != mat2shape[1])) throw std::invalid_argument("Matrices are not of the same shape!");

	int size = matrix1->returnSize();
	int bytes = size * sizeof(float);

	std::unique_ptr<float[]> mat1 = matrix1->returnMatrix();
	std::unique_ptr<float[]> mat2 = matrix2->returnMatrix();
	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(bytes);

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, bytes);
	cudaMalloc(&mat2d, bytes);
	cudaMalloc(&mat3d, bytes);
	cudaMemcpy(mat1d, mat1.get(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2.get(), bytes, cudaMemcpyHostToDevice);

	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;

	addD <<< NUM_BLOCKS, NUM_THREADS >>> (size, mat1d, mat2d, mat3d);

	cudaMemcpy(mat3.get(), mat3d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<int[]> shape = matrix1->returnShape();
	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, shape);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

template <typename Lambda>
__global__ 
void apply(int size, Lambda func, float* toApply) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) toApply[idx] = func(toApply[idx]);
}

int main() {
	std::unique_ptr<int[]> shape1 = std::make_unique<int[]>(2);
	shape1[0] = 5;
	shape1[1] = 2;
	std::unique_ptr<float[]> vals1 = std::make_unique<float[]>(10);
	for (int i = 0; i < 10; i++) {
		vals1[i] = 1.0f;
	}
	std::unique_ptr<Matrix> matrix1 = std::make_unique<Matrix>(vals1, shape1);

	std::unique_ptr<int[]> shape2 = std::make_unique<int[]>(2);
	shape2[0] = 5;
	shape2[1] = 2;
	std::unique_ptr<float[]> vals2 = std::make_unique<float[]>(10);
	for (int i = 0; i < 10; i++) {
		vals2[i] = 1.0f;
	}
	std::unique_ptr<Matrix> matrix2 = std::make_unique<Matrix>(vals2, shape2);

	std::unique_ptr<Matrix> added = addMatrices(matrix1, matrix2);


	float* memCpy;
	cudaMalloc(&memCpy, 10 * sizeof(float));
	cudaMemcpy(memCpy, added.get(), 10*sizeof(float), cudaMemcpyHostToDevice);

	auto makeNegative = [=] __host__ __device__(float x) { return -1 * x; };

	apply <<< (10 + 256 - 1)/256, 256 >>> (10, makeNegative, memCpy);

	std::unique_ptr<float[]> copyPtr = std::make_unique<float[]>(10);
	cudaMemcpy(copyPtr.get(), memCpy, 10*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(memCpy);
}

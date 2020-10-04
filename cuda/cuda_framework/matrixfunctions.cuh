#pragma once
#include "matrix.cuh"

template <typename Lambda>
std::unique_ptr<Matrix> apply(std::unique_ptr<Matrix>& in_matrix, Lambda function);
std::unique_ptr<Matrix> add(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);
std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);
std::unique_ptr<Matrix> genRand(int rows, int cols);

template <typename Lambda>
__global__
void applyD(int size, float* inVector, Lambda function) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) inVector[index] = function(inVector[index]);
}

template <typename Lambda>
std::unique_ptr<Matrix> apply(std::unique_ptr<Matrix>& in_matrix, Lambda function) {
	std::unique_ptr<float[]> matrix = in_matrix->returnMatrix();
	int size = in_matrix->returnSize();
	std::unique_ptr<int[]> shape = in_matrix->returnShape();

	int bytes = size * sizeof(float);

	float* dCopy;
	cudaMalloc(&dCopy, bytes);
	cudaMemcpy(dCopy, matrix.get(), bytes, cudaMemcpyHostToDevice);

	GPUParams gpu;
	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
	applyD <<< dimGridX, gpu.THREAD_SIZE >>> (size, dCopy, function);

	std::unique_ptr<float[]> new_matrix = std::make_unique<float[]>(size);
	cudaMemcpy(new_matrix.get(), dCopy, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(new_matrix, shape);

	cudaFree(dCopy);

	return ret_matrix;
}


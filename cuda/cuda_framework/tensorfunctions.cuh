#pragma once
#include "tensor.cuh"
#include <cuda_runtime.h>

struct GPU {
	int THREAD_SIZE = 1 << 10;
	int BLOCK_SIZE = 1 << 5;
};

std::unique_ptr<Tensor> reshape(std::unique_ptr<Tensor>& tensor, int new_depth, int new_rows, int new_cols);
std::unique_ptr<Tensor> transpose(std::unique_ptr<Tensor>& tensor);
std::unique_ptr<Tensor> add(std::unique_ptr<Tensor>& tensor1, std::unique_ptr<Tensor>& tensor2);
std::unique_ptr<Tensor> subtract(std::unique_ptr<Tensor>& tensor1, std::unique_ptr<Tensor>& tensor2);

template <typename Lambda>
__global__
void applyD(int size, float* inVector, Lambda function) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) inVector[index] = function(inVector[index]);
}

template <typename Lambda>
std::unique_ptr<Tensor> apply(std::unique_ptr<Tensor>& tensor, Lambda function) {
	int size = tensor->returnTensorSize();
	int bytes = size * sizeof(float);

	float* t1d;
	cudaMalloc(&t1d, bytes);

	std::unique_ptr<float[]> t1 = tensor->returnTensor();
	cudaMemcpy(t1d, t1.get(), bytes, cudaMemcpyHostToDevice);

	GPU gpu;
	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
	applyD <<< dimGridX, gpu.THREAD_SIZE >>> (size, t1d, function);

	std::unique_ptr<float[]> new_tensor(new float[size]);
	cudaMemcpy(new_tensor.get(), t1d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<int[]> shape = tensor->returnShape();
	std::unique_ptr<Tensor> ret_tensor(new Tensor(new_tensor, shape));

	cudaFree(dCopy);

	return ret_matrix;
}

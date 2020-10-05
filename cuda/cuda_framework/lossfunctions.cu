#include "lossfunctions.cuh"

std::unique_ptr<Matrix> mse(std::unique_ptr<Matrix>& predicted, std::unique_ptr<Matrix>& actual) {
	std::unique_ptr<Matrix> ret_matrix = subtract(predicted, actual);

	return ret_matrix;
}

__global__
void crossentropyD(int size, float* predicted, float* actual, float* retMatrix) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) retMatrix[index] = -1 * (actual[index] / predicted[index]) + (1 - actual[index]) / (1 - predicted[index]);
}

std::unique_ptr<Matrix> crossentropy(std::unique_ptr<Matrix>& predicted, std::unique_ptr<Matrix>& actual) {
	std::unique_ptr<int[]> shape1 = predicted->returnShape();
	std::unique_ptr<int[]> shape2 = actual->returnShape();
	if ((shape1[0] != shape2[0]) || (shape1[1] != shape2[1])) throw std::invalid_argument("Matrices are not of same dimensions!");

	int size = predicted->returnSize();
	int bytes = size * sizeof(float);

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, bytes);
	cudaMalloc(&mat2d, bytes);
	cudaMalloc(&mat3d, bytes);

	std::unique_ptr<float[]> mat1 = predicted->returnMatrix();
	std::unique_ptr<float[]> mat2 = actual->returnMatrix();

	cudaMemcpy(mat1d, mat1.get(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2.get(), bytes, cudaMemcpyHostToDevice);

	GPUParams gpu;
	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
	crossentropyD <<< dimGridX, gpu.THREAD_SIZE >>> (size, mat1d, mat2d, mat3d);

	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(size);
	cudaMemcpy(mat3.get(), mat3d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, shape1);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

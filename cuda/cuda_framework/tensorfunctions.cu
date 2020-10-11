#include "tensorfunctions.cuh"

std::unique_ptr<Tensor> reshape(std::unique_ptr<Tensor>& tensor, int new_depth, int new_rows, int new_cols) {
	if (new_depth * new_rows * new_cols != tensor->returnTensorSize()) throw std::invalid_argument("New dimensions do not match old size!");
	std::unique_ptr<float[]> tensor_raw = tensor->returnTensor();
	std::unique_ptr<int[]> shape(new int[3]{ new_depth, new_rows, new_cols });
	std::unique_ptr<Tensor> ret_tensor(new Tensor(tensor_raw, shape));

	return ret_tensor;
}

__global__
void transposeD(int depths, int rows, int cols, float* inVector) {
	int depth = blockIdx.z * blockDim.z + threadIdx.z;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < rows) && (col < cols) && (depth < depths)) inVector[depth * rows * cols + col * rows + row] = inVector[depth * rows * cols + row * cols + col];
}

// The transpose I could probably also do a full rotation of the matrix or I could just not do that
std::unique_ptr<Tensor> transpose(std::unique_ptr<Tensor>& tensor) {
	std::unique_ptr<int[]> old_shape = tensor->returnShape();
	std::unique_ptr<int[]> new_shape(new int[3]{ old_shape[0], old_shape[2], old_shape[1] });

	int size = tensor->returnTensorSize();
	int bytes = size * sizeof(float);

	float* t1d;
	cudaMalloc(&t1d, bytes);

	std::unique_ptr<float[]> t1 = tensor->returnTensor();
	cudaMemcpy(t1d, t1.get(), bytes, cudaMemcpyHostToDevice);

	GPU gpu;
	int grid_depth = (old_shape[0] + gpu.BLOCK_SIZE - 1) / gpu.BLOCK_SIZE;
	int grid_rows = (old_shape[1] + gpu.BLOCK_SIZE - 1) / gpu.BLOCK_SIZE;
	int grid_cols = (old_shape[2] + gpu.BLOCK_SIZE - 1) / gpu.BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows, grid_depth);
	dim3 dimBlock(gpu.BLOCK_SIZE, gpu.BLOCK_SIZE, gpu.BLOCK_SIZE);

	transposeD <<< dimGrid, dimBlock >>> (old_shape[0], old_shape[1], old_shape[2], t1d);

	std::unique_ptr<float[]> new_tensor(new float[size]);
	cudaMemcpy(new_tensor.get(), t1d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Tensor> ret_tensor(new Tensor(new_tensor, new_shape));

	cudaFree(t1d);

	return ret_tensor;
}

// Now I just have to add the extra functions and then go and modify everything based around these tensors and such

//__global__
//void addD(int size, float* vector1, float* vector2, float* retVector) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	if (index < size) retVector[index] = vector1[index] + vector2[index];
//}
//
//std::unique_ptr<Tensor> add(std::unique_ptr<Tensor>& tensor1, std::unique_ptr<Tensor>& tensor2) {
//	std::unique_ptr<int[]> shape1 = tensor1->returnShape();
//	std::unique_ptr<int[]> shape2 = tensor2->returnShape();
//	if ((shape1[0] != shape2[0]) || (shape1[1] != shape2[1]) || (shape1[2] != shape2[2])) throw std::invalid_argument("Tensors are not of same shape!");
//
//	int size = tensor1->returnTensorSize();
//	int bytes = size * sizeof(float);
//
//	std::unique_ptr<float[]> t1 = tensor1->returnTensor();
//	std::unique_ptr<float[]> t2 = tensor2->returnTensor();
//
//	float* t1d;
//	float* t2d;
//	float* t3d;
//	cudaMalloc(&t1d, bytes);
//	cudaMalloc(&t2d, bytes);
//	cudaMalloc(&t3d, bytes);
//	cudaMemcpy(t1d, t1.get(), bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(t2d, t2.get(), bytes, cudaMemcpyHostToDevice);
//
//	GPU gpu;
//	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
//	addD <<< dimGridX, gpu.THREAD_SIZE >>> (size, t1d, t2d, t3d);
//
//	std::unique_ptr<float[]> t3(new float[size]);
//	cudaMemcpy(t3.get(), t3d, bytes, cudaMemcpyDeviceToHost);
//
//	std::unique_ptr<int[]> shape = tensor1->returnShape();
//	std::unique_ptr<Tensor> ret_matrix(new Tensor(t3, shape));
//
//	cudaFree(t1d);
//	cudaFree(t2d);
//	cudaFree(t3d);
//
//	return ret_matrix;
//}
//
//__global__
//void subtractD(int size, float* vector1, float* vector2, float* retVector) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	if (index < size) retVector[index] = vector1[index] - vector2[index];
//}
//
//std::unique_ptr<Tensor> subtract(std::unique_ptr<Tensor>& tensor1, std::unique_ptr<Tensor>& tensor2) {
//	std::unique_ptr<int[]> shape1 = tensor1->returnShape();
//	std::unique_ptr<int[]> shape2 = tensor2->returnShape();
//	if ((shape1[0] != shape2[0]) || (shape1[1] != shape2[1]) || (shape1[2] != shape2[2])) throw std::invalid_argument("Tensors are not of same shape!");
//
//	int size = tensor1->returnTensorSize();
//	int bytes = size * sizeof(float);
//
//	std::unique_ptr<float[]> t1 = tensor1->returnTensor();
//	std::unique_ptr<float[]> t2 = tensor2->returnTensor();
//
//	float* t1d;
//	float* t2d;
//	float* t3d;
//	cudaMalloc(&t1d, bytes);
//	cudaMalloc(&t2d, bytes);
//	cudaMalloc(&t3d, bytes);
//	cudaMemcpy(t1d, t1.get(), bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(t2d, t2.get(), bytes, cudaMemcpyHostToDevice);
//
//	GPU gpu;
//	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
//	subtractD <<< dimGridX, gpu.THREAD_SIZE >>> (size, t1d, t2d, t3d);
//
//	std::unique_ptr<float[]> t3(new float[size]);
//	cudaMemcpy(t3.get(), t3d, bytes, cudaMemcpyDeviceToHost);
//
//	std::unique_ptr<int[]> shape = tensor1->returnShape();
//	std::unique_ptr<Tensor> ret_matrix(new Tensor(t3, shape));
//
//	cudaFree(t1d);
//	cudaFree(t2d);
//	cudaFree(t3d);
//
//	return ret_matrix;
//}
//

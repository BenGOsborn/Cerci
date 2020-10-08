#include "matrixfunctions.cuh"

// Defining the constants section for the thread blocks is a bit of a pain and feels messy

__global__
void addD(int size, float* vector1, float* vector2, float* retVector) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) retVector[index] = vector1[index] + vector2[index];
}

std::unique_ptr<Matrix> add(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2) {
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

	GPUParams gpu;
	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
	addD <<< dimGridX, gpu.THREAD_SIZE >>> (size, mat1d, mat2d, mat3d);

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
void multiplyD(int rows, int same, int cols, float* vector1, float* vector2, float* retVector) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rows) && (col < cols)) {
		float sum = 0;
		for (int i = 0; i < same; i++) {
			sum += vector1[row * same + i] * vector2[i * cols + col];
		}
		retVector[row * cols + col] = sum;
	}
}

std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2) {
	std::unique_ptr<int[]> mat1shape = matrix1->returnShape();
	std::unique_ptr<int[]> mat2shape = matrix2->returnShape();
	if (mat1shape[1] != mat2shape[0]) throw std::invalid_argument("Matrix1's cols must equal Matrix2's rows!");

	std::unique_ptr<int[]> new_shape = std::make_unique<int[]>(2);
	new_shape[0] = mat1shape[0];
	new_shape[1] = mat2shape[1];
	int same = mat1shape[1];

	int mat1bytes = matrix1->returnSize() * sizeof(float);
	int mat2bytes = matrix2->returnSize() * sizeof(float);
	int mat3bytes = new_shape[0] * new_shape[1] * sizeof(float);

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

	GPUParams gpu;
	int grid_rows = (new_shape[0] + gpu.BLOCK_SIZE - 1) / gpu.BLOCK_SIZE;
	int grid_cols = (new_shape[1] + gpu.BLOCK_SIZE - 1) / gpu.BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(gpu.BLOCK_SIZE, gpu.BLOCK_SIZE);

	multiplyD <<< dimGrid, dimBlock >>> (new_shape[0], same, new_shape[1], mat1d, mat2d, mat3d);

	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(new_shape[0] * new_shape[1]);
	cudaMemcpy(mat3.get(), mat3d, mat3bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, new_shape);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

std::unique_ptr<Matrix> genRand(int rows, int cols) {
	std::unique_ptr<int[]> shape = std::make_unique<int[]>(2);
	shape[0] = rows;
	shape[1] = cols;
	int size = rows * cols;

	std::unique_ptr<float[]> vals = std::make_unique<float[]>(size);

	float randVal = 0.0f;
	for (int i = 0; i < size; i++) {
		if (rand() % 10 > 5) {
			randVal = 1.0 * (std::rand() % 100) / 100;
		}
		else {
			randVal = -1.0 * (std::rand() % 100) / 100;
		}
		vals[i] = randVal;
	}

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(vals, shape);

	return ret_matrix;
}

std::unique_ptr<Matrix> genZeros(int rows, int cols) {
	std::unique_ptr<int[]> shape = std::make_unique<int[]>(2);
	shape[0] = rows;
	shape[1] = cols;
	int size = rows * cols;

	std::unique_ptr<float[]> vals = std::make_unique<float[]>(size);
	for (int i = 0; i < size; i++) {
		vals[i] = 0.0f;
	}

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(vals, shape);

	return ret_matrix;
}

float sum(std::unique_ptr<Matrix>& matrix) {
	std::unique_ptr<float[]> returned_matrix = matrix->returnMatrix();
	int size = matrix->returnSize();

	float sm = 0;
	for (int i = 0; i < size; i++) {
		sm += returned_matrix[i];
	}

	return sm;
}

__global__
void multiplyAllD(int size, float* vector1, float* vector2, float* retVector) {
	int index = blockIdx.y * blockDim.y + threadIdx.y;

	if (index < size) retVector[index] = vector1[index] * vector2[index];
}

std::unique_ptr<Matrix> multiplyElementwise(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2) {
	std::unique_ptr<int[]> shape1 = matrix1->returnShape();
	std::unique_ptr<int[]> shape2 = matrix2->returnShape();
	if ((shape1[0] != shape2[0]) || (shape1[1] != shape2[1])) throw std::invalid_argument("Dimensions of matrices are not the same!");

	int size = matrix1->returnSize();
	int bytes = size * sizeof(float);

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, bytes);
	cudaMalloc(&mat2d, bytes);
	cudaMalloc(&mat3d, bytes);

	std::unique_ptr<float[]> mat1 = matrix1->returnMatrix();
	std::unique_ptr<float[]> mat2 = matrix2->returnMatrix();

	cudaMemcpy(mat1d, mat1.get(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2.get(), bytes, cudaMemcpyHostToDevice);

	GPUParams gpu;
	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
	multiplyAllD <<< dimGridX, gpu.THREAD_SIZE >>> (size, mat1d, mat2d, mat3d);

	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(size);
	cudaMemcpy(mat3.get(), mat3d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, shape1);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

__global__
void divideAllD(int size, float* vector1, float* vector2, float* retVector) {
	int index = blockIdx.y * blockDim.y + threadIdx.y;

	if (index < size) retVector[index] = vector1[index] / vector2[index];
}

std::unique_ptr<Matrix> divideElementwise(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2) {
	std::unique_ptr<int[]> shape1 = matrix1->returnShape();
	std::unique_ptr<int[]> shape2 = matrix2->returnShape();
	if ((shape1[0] != shape2[0]) || (shape1[1] != shape2[1])) throw std::invalid_argument("Dimensions of matrices are not the same!");

	int size = matrix1->returnSize();
	int bytes = size * sizeof(float);

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, bytes);
	cudaMalloc(&mat2d, bytes);
	cudaMalloc(&mat3d, bytes);

	std::unique_ptr<float[]> mat1 = matrix1->returnMatrix();
	std::unique_ptr<float[]> mat2 = matrix2->returnMatrix();

	cudaMemcpy(mat1d, mat1.get(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2.get(), bytes, cudaMemcpyHostToDevice);

	GPUParams gpu;
	int dimGridX = (size + gpu.THREAD_SIZE - 1) / gpu.THREAD_SIZE;
	divideAllD <<< dimGridX, gpu.THREAD_SIZE >>> (size, mat1d, mat2d, mat3d);

	std::unique_ptr<float[]> mat3 = std::make_unique<float[]>(size);
	cudaMemcpy(mat3.get(), mat3d, bytes, cudaMemcpyDeviceToHost);

	std::unique_ptr<Matrix> ret_matrix = std::make_unique<Matrix>(mat3, shape1);

	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);

	return ret_matrix;
}

std::unique_ptr<Matrix> subtract(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2) {
	auto make_negative = [] __host__ __device__(float x) { return -1 * x; };
	std::unique_ptr<Matrix> negative = apply(matrix2, make_negative);

	std::unique_ptr<Matrix> subtracted = add(matrix1, negative);

	return subtracted;
}

std::unique_ptr<Matrix> multiplyScalar(std::unique_ptr<Matrix>& matrix, float val) {
	auto do_multiply = [=] __host__ __device__(float x) { return val * x; };
	std::unique_ptr<Matrix> multiplied = apply(matrix, do_multiply);

	return multiplied;
}

std::unique_ptr<Matrix> divideScalar(std::unique_ptr<Matrix>& matrix, float val) {
	auto do_multiply = [=] __host__ __device__(float x) { return x / val; };
	std::unique_ptr<Matrix> divided = apply(matrix, do_multiply);

	return divided;
}

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>

class Matrix {
	private:
		float* matrix;
		unsigned int* size;
		unsigned int* shape;

	public:
		// This shape and matrix being parsed seperaretly will probably have to be changed at some point in time or we can just keep it like this?
		Matrix(float *inMatrix, unsigned int *inShape) {

			shape = (unsigned int*)malloc(3 * sizeof(unsigned int));
			memcpy(shape, inShape, 3 * sizeof(unsigned int));

			size = new unsigned int(shape[0]*shape[1]);
			
			matrix = (float*)malloc(*size * sizeof(float));
			memcpy(matrix, inMatrix, *size * sizeof(float));
		}

		void print() {
			for (int i = 0; i < shape[0]; i++) {
				for (int j = 0; j < shape[1]; j++) {
					std::cout << matrix[i*shape[1]+j] << " ";
				}
				std::cout << "\n";
			}
		}

		Matrix* reshape(unsigned int rows, unsigned int cols) {
			if (rows * cols != *size) throw std::invalid_argument("New matrix size does not match original matrix size!");

			unsigned int* new_shape;
			new_shape = (unsigned int*)malloc(2 * sizeof(unsigned int));
			new_shape[0] = rows;
			new_shape[1] = cols;

			Matrix* ret_matrix = new Matrix(matrix, new_shape);

			free(new_shape);

			return ret_matrix;
		}

		// This could be done in parallel on the GPU
		Matrix* transpose() {
			unsigned int* new_shape;
			new_shape = (unsigned int*)malloc(2 * sizeof(unsigned int));	
			new_shape[0] = shape[1];
			new_shape[1] = shape[0];

			float* new_matrix;
			new_matrix = (float*)malloc(*size * sizeof(float));
			for (int i = 0; i < shape[0]; i++) {
				for (int j = 0; j < shape[1]; j++) {
					new_matrix[j * new_shape[1] + i] = matrix[i * shape[1] + j];
				}
			}

			Matrix* ret_matrix = new Matrix(new_matrix, new_shape);
			
			free(new_shape);
			free(new_matrix);

			return ret_matrix;
		}

		Matrix* clone() {
			Matrix* ret_matrix = new Matrix(matrix, shape);
			return ret_matrix;
		}

		// Functions like this can probably be done in parallel on the GPU
		Matrix* applyFunc(float(*func)(float)) {
			float* new_matrix;
			new_matrix = (float*)malloc(*size * sizeof(float));

			for (int i = 0; i < *size; i++) {
				new_matrix[i] = func(new_matrix[i]);
			}

			Matrix* ret_matrix = new Matrix(new_matrix, shape);

			free(new_matrix);

			return ret_matrix;
		}

		float* returnMatrix() {
			float* ret_matrix;
			ret_matrix = (float*)malloc(*size * sizeof(float));
			memcpy(ret_matrix, matrix, *size * sizeof(float));

			return ret_matrix;
		}

		unsigned int* returnShape() {
			return shape;
		}

		unsigned int returnSize() {
			return *size;
		}

		~Matrix() {
			free(matrix);
			free(size);
			free(shape);
		}
};

__global__
void addMatricesD(int size, float* vector1, float *vector2, float *retVector) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) retVector[idx] = vector1[idx] + vector2[idx];
}

// For subtraction just multiply the array by 1 the add into a negative which can be done through the apply func
Matrix* addMatrices(Matrix *matrix1, Matrix *matrix2) {
	if (matrix1->returnShape() != matrix2->returnShape()) throw std::invalid_argument("Matrices are not of the same shape!");

	unsigned int size = matrix1->returnSize();
	int bytes = size * sizeof(float);

	float* mat1 = matrix1->returnMatrix();
	float* mat2 = matrix2->returnMatrix();
	float* mat3;
	mat3 = (float*)malloc(bytes);

	float* mat1d;
	float* mat2d;
	float* mat3d;
	cudaMalloc(&mat1d, bytes);
	cudaMalloc(&mat2d, bytes);
	cudaMalloc(&mat3d, bytes);
	cudaMemcpy(mat1d, mat1, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2d, mat2, bytes, cudaMemcpyHostToDevice);

	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;

	addVectors <<< NUM_BLOCKS, NUM_THREADS >>> (size, mat1d, mat2d, mat3d);

	cudaMemcpy(mat3, mat3d, bytes, cudaMemcpyDeviceToHost);

	unsigned int* shape = matrix1->returnShape();
	Matrix* ret_matrix = new Matrix(mat3, shape);

	// What pointers are the ones that I need to free up here?
	cudaFree(mat1d);
	cudaFree(mat2d);
	cudaFree(mat3d);
	free(shape);
	free(mat1);
	free(mat2);
	free(mat3);

	return ret_matrix;
}

__global__
void multiplyMatricesD() {
	
}

Matrix* multiplyMatrices(Matrix *matrix1, Matrix *matrix2) {
	if (matrix1->returnShape()[1] != matrix2->returnShape()[0]) throw std::invalid_argument("Matrix dimensions are not aligned for multiplication!");

	
}

int main() {
	unsigned int* shape;
	shape = (unsigned int*)malloc(2 * sizeof(unsigned int));
	shape[0] = 5;
	shape[1] = 2;

	float* vals;
	vals = (float*)malloc(10 * sizeof(float));
	for (int i = 0; i < 10; i++) {
		vals[i] = 1.0f;
	}

	Matrix* matrix = new Matrix(vals, shape);
	Matrix* transposed = matrix->transpose();
	transposed->print();

	delete matrix;
	delete transposed;
	free(shape);
	free(vals);
}
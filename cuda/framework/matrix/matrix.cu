#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>

/// Can I potentially turn this into a tensor class which supports both matrices and tensors in the same class?
class Matrix {
	public:
		float* matrix;
		int* size;
		int* shape;

		// This shape and matrix being parsed seperaretly will probably have to be changed at some point in time or we can just keep it like this?
		Matrix(float *inMatrix, int *inShape) {
			shape = (int*)malloc(2 * sizeof(int));
			memcpy(shape, inShape, 2*sizeof(int));

			size = new int(shape[0]*shape[1]);

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

		Matrix* reshape(int rows, int cols) {
			if (rows * cols != *size) throw new std::invalid_argument("New matrix size does not match original matrix size!");

			int* new_shape;
			new_shape = (int*)malloc(2 * sizeof(int));
			new_shape[0] = rows;
			new_shape[1] = cols;

			Matrix* ret_matrix = new Matrix(matrix, new_shape);

			free(new_shape);

			return ret_matrix;
		}

		// This could be done in parallel on the GPU
		Matrix* transpose() {
			int* new_shape;
			new_shape = (int*)malloc(2 * sizeof(int));	
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

		int* shape() {
			return shape;
		}

		~Matrix() {
			free(matrix);
			free(size);
			free(shape);
		}
};

int main() {

	int* shape;
	shape = (int*)malloc(2 * sizeof(int));
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
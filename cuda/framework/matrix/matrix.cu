#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

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
		vals[i] = 1;
	}

	Matrix* matrix = new Matrix(vals, shape);
}
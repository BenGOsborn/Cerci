#include <cuda_runtime.h>
#include <stdlib.h>

// Is it completely necessary to make a brand new copy everytime we want to copy the matrix?
// Could we just copy the values into the array instead?
__global__
void copyMatrix(int N, float readmatrix, float writeMatrix) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) writeMatrix[i] = readMatrix[i];
}

/// Can I potentially turn this into a tensor class which supports both matrices and tensors in the same class?
class Matrix {
	public:
		float* matrix;
		int* size;
		int* shape[2];

		Matrix(float *inMatrix, int *inShape) {
				// This should point to the same array as the shape parsed in
				*shape = inShape;
				*size = inShape[0] * inShape[1];
				matrix = (float*)malloc(*size);

				// Can I make use of the GPU to copy the values aswell?
				float* rMatrix, * wMatrix;
				cudaMalloc(&rMatrix, size);
				cudaMalloc(&wMatrix, size);
				cudaMemcpy(rMatrix, inShape, *size, cudaMemcpyHostToDevice);

				// Now I just need to specify the block and the thread amounts somehow...?
				copyMatrix <<< (*size + 255) / 256, 256 >>> (*size, rMatrix, wMatrix);
				
				cudaMemcpy(matrix, wMatrix, *size, cudaMemcpyDeviceToHost);

				cudaFree(rMatrix);
				cudaFree(wMatrix);

				// Should I also destroy the input matrix here since its already created a copy of it?
			}
};

int main() {


}
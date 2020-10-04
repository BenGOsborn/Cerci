#pragma once
// Clean these up eventually
// Change variable name type (underscores)

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>

// Maybe get rid of this and replace it with a define statement is that possible ?
struct GPUParams {
	int THREAD_SIZE = 1 << 10;
	int BLOCK_SIZE = 1 << 5;
};

class Matrix {
private:
	std::unique_ptr<float[]> matrix;
	std::unique_ptr<int> size;
	std::unique_ptr<int[]> shape;
	std::unique_ptr<GPUParams> gpu;
public:
	Matrix(std::unique_ptr<float[]>& inMatrix, std::unique_ptr<int[]>& inShape);
	void print();
	std::unique_ptr<Matrix> reshape(int rows, int cols);
	std::unique_ptr<Matrix> transpose();
	std::unique_ptr<Matrix> clone();
	std::unique_ptr<float[]> returnMatrix();
	std::unique_ptr<int[]> returnShape();
	int returnSize();
};

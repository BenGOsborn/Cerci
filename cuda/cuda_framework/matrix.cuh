#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

struct Constants {
	int THREAD_SIZE = 1 << 10;
	int BLOCK_SIZE = 1 << 5;
};

class Matrix {
private:
	std::unique_ptr<float[]> matrix;
	std::unique_ptr<int> size;
	std::unique_ptr<int[]> shape;
public:
	Matrix(std::unique_ptr<float[]>& inMatrix, std::unique_ptr<int[]>& inShape);
	void print();
	std::unique_ptr<Matrix> reshape(int rows, int cols);
	std::unique_ptr<Matrix> transpose();
	std::unique_ptr<Matrix> apply(float(*function)(float));
	std::unique_ptr<Matrix> clone();
	std::unique_ptr<float[]> returnMatrix();
	std::unique_ptr<int[]> returnShape();
	int returnSize();
};

std::unique_ptr<Matrix> add(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);

std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);

std::unique_ptr<Matrix> genRand(int rows, int cols);

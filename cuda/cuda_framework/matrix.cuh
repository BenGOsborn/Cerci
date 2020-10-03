#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

template <typename Lambda>
__global__ void applyD(int size, float* inVector, Lambda function);

__global__ void transposeD(int rows, int cols, float* inVector);

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
	std::unique_ptr<Matrix> clone();
	template <typename Lambda>
	std::unique_ptr<Matrix> apply(Lambda function);
	std::unique_ptr<float[]> returnMatrix();
	std::unique_ptr<int[]> returnShape();
	int returnSize();
};

__global__ void addD(int size, float* vector1, float* vector2, float* retVector);
std::unique_ptr<Matrix> add(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);

__global__ void multiplyD(int rows, int same, int cols, float* vector1, float* vector2, float* retVector);
std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);

std::unique_ptr<Matrix> genRand(int rows, int cols);

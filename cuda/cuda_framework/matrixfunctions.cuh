#pragma once
#include "matrix.cuh"

template <typename Lambda>
std::unique_ptr<Matrix> apply(Lambda function);
std::unique_ptr<Matrix> add(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);
std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix>& matrix1, std::unique_ptr<Matrix>& matrix2);
std::unique_ptr<Matrix> genRand(int rows, int cols);

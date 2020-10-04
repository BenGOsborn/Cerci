#pragma once

#include "matrixfunctions.cuh"
#include <math.h>

// I have to make sure that CUDA will work with the math.h function
std::unique_ptr<Matrix> applySigmoid(std::unique_ptr<Matrix>& matrix, bool deriv = false);
std::unique_ptr<Matrix> applyLRelu(std::unique_ptr<Matrix>& matrix, bool deriv = false);
std::unique_ptr<Matrix> applySoftmax(std::unique_ptr<Matrix>& matrix, bool deriv = false);
std::unique_ptr<Matrix> applyTanh(std::unique_ptr<Matrix>& matrix, bool deriv = false);

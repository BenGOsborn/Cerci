#pragma once

#include "matrixfunctions.cuh"
#include <math.h>

// Extra params goes here just for the softmax function
std::unique_ptr<Matrix> sigmoid(std::unique_ptr<Matrix>& matrix, bool deriv = false);
std::unique_ptr<Matrix> lRelu(std::unique_ptr<Matrix>& matrix, bool deriv = false);
std::unique_ptr<Matrix> softmax(std::unique_ptr<Matrix>& matrix, bool deriv = false);


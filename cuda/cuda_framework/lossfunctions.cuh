#pragma once

#include "matrixfunctions.cuh"

std::unique_ptr<Matrix> mse(std::unique_ptr<Matrix>& predicted, std::unique_ptr<Matrix>& actual);
std::unique_ptr<Matrix> crossentropy(std::unique_ptr<Matrix>& predicted, std::unique_ptr<Matrix>& actual);
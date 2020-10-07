#pragma once

#include "matrixfunctions.cuh"
#include <math.h>

class Sigmoid {
	Sigmoid();
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& sigmoid, std::unique_ptr<Matrix>& errors);
};

class LRelu {
	LRelu();
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& lrelu, std::unique_ptr<Matrix>& errors);
};

class Softmax {
	Softmax();
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& softmax, std::unique_ptr<Matrix>& errors);
};

class Tanh {
	Tanh();
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& tanh, std::unique_ptr<Matrix>& errors);
};

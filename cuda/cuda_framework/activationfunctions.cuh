#pragma once

#include "matrixfunctions.cuh"
#include <math.h>

// We want to store all of the values within the state of the network for backprop
class Sigmoid {
protected:
	std::unique_ptr<Matrix> hidden_layer;
public:
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& errors);
};

class LRelu {
protected:
	std::unique_ptr<Matrix> hidden_layer;
public:
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& errors);
};

class Softmax {
protected:
	std::unique_ptr<Matrix> hidden_layer;
public:
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& errors);
};

class Tanh {
protected:
	std::unique_ptr<Matrix> hidden_layer;
public:
	std::unique_ptr<Matrix> forwardProp(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backProp(std::unique_ptr<Matrix>& errors);
};

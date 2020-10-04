#pragma once
#include "matrix.cuh"
#include <functional>

// How am I going to define the activation function I want to use with a lambda?
class FullyConnected {
private:
	std::unique_ptr<Matrix> weights;
	std::unique_ptr<Matrix> bias;
	float(*activation_func)(float);
public:
	FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float(*function)(float));
	std::unique_ptr<Matrix> predict(std::unique_ptr<Matrix> &inputs);
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix> &inputs, std::unique_ptr<Matrix> &predicted, std::unique_ptr<Matrix> &errors_raw);
	// There isnt a very good way to deal with a returnNetwork return type hey
};
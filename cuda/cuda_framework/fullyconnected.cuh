#pragma once
#include "activationfunctions.cuh"

struct Network {
	std::unique_ptr<Matrix> weights;
	std::unique_ptr<Matrix> bias;
	Network(std::unique_ptr<Matrix>& in_weights, std::unique_ptr<Matrix>& in_bias) {
		weights = in_weights->clone();
		bias = in_bias->clone();
	}
};

class FullyConnected {
private:
	int activation_function;
	std::unique_ptr<Network> network;
public:
	FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, std::string activation_func);
	std::unique_ptr<Matrix> predict(std::unique_ptr<Matrix> &inputs);
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix> &inputs, std::unique_ptr<Matrix> &predicted, std::unique_ptr<Matrix> &errors_raw);
	std::unique_ptr<Network> returnNetwork();
};

FullyConnected::FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, std::string activation_func) {
	// I want to make it so it can support any function at some point
	if (activation_func.compare("sigmoid") == 0) {
		FullyConnected::activation_function = 0;
	}
	else if (activation_func.compare("lrelu") == 0) {
		FullyConnected::activation_function = 1;
	}
	else if (activation_func.compare("softmax") == 0) {
		FullyConnected::activation_function = 2;
	}
	else if (activation_func.compare("tanh") == 0) {
		FullyConnected::activation_function = 3;
	} else {
		throw std::invalid_argument("Invalid activation function!");
	}
	FullyConnected::network = std::make_unique<Network>(weight_set, bias_set);
}
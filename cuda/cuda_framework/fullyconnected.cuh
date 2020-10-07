#pragma once
#include "activationfunctions.cuh"
// These includes need to get cleaned up

// This network is just going to output the raw without the activation function of course
// Each part will be in its own seperate class and will back propagate properly
class FullyConnected {
protected:
	std::unique_ptr<Matrix> weights;
	std::unique_ptr<Matrix> bias;
	float learning_rate;
public:
	FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1);
	std::unique_ptr<Matrix> predict(std::unique_ptr<Matrix>& inputs);
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors);
	void setNetwork(std::unique_ptr<Matrix>& new_weights, std::unique_ptr<Matrix>& new_bias);
};

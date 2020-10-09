#pragma once
#include "activationfunctions.cuh"
// These includes need to get cleaned up

// This network is just going to output the raw without the activation function of course
// Each part will be in its own seperate class and will back propagate properly
class FullyConnected {
public:
	// We probably want to store the weights here too
	std::unique_ptr<Matrix> weights;
	std::unique_ptr<Matrix> bias;
	float learning_rate;

	std::unique_ptr<Matrix> hidden_layer;
public:
	FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1);
	std::unique_ptr<Matrix> predict(std::unique_ptr<Matrix>& inputs);
	virtual std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& errors);
	void setNetwork(std::unique_ptr<Matrix>& new_weights, std::unique_ptr<Matrix>& new_bias);
};

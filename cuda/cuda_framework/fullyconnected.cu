#include "fullyconnected.cuh"

FullyConnected::FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1) {
	FullyConnected::weights = weight_set->clone();
	FullyConnected::bias = bias_set->clone();
	FullyConnected::learning_rate = lr;
}

std::unique_ptr<Matrix> FullyConnected::predict(std::unique_ptr<Matrix>& inputs) {
	std::unique_ptr<Matrix> multiplied = multiply(FullyConnected::weights, inputs);
	std::unique_ptr<Matrix> out = add(multiplied, FullyConnected::bias);

	return out;
}

std::unique_ptr<Matrix> FullyConnected::train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors) {
	std::unique_ptr<Matrix> inputs_transposed = inputs->transpose();
	std::unique_ptr<Matrix> weight_adjustments = multiply(errors, inputs_transposed);

	std::unique_ptr<Matrix> weights_lr = multiplyScalar(weight_adjustments, FullyConnected::learning_rate);
	std::unique_ptr<Matrix> bias_lr = multiplyScalar(errors, FullyConnected::learning_rate);

	FullyConnected::weights = subtract(FullyConnected::weights, weights_lr);
	FullyConnected::bias = subtract(FullyConnected::bias, bias_lr);

	std::unique_ptr<Matrix> weights_transposed = FullyConnected::weights->transpose();
	std::unique_ptr<Matrix> back_errors = multiply(errors, weights_transposed);

	return back_errors;
}

// This function should only ever be used for deepQlearning where two models are required
void FullyConnected::setNetwork(std::unique_ptr<Matrix>& new_weights, std::unique_ptr<Matrix>& new_bias) {
	FullyConnected::weights = new_weights->clone();
	FullyConnected::bias = new_bias->clone();
}

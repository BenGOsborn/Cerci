#include "fullyconnected.cuh"

FullyConnected::FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr) {
	FullyConnected::weights = weight_set->clone();
	FullyConnected::bias = bias_set->clone();
	FullyConnected::learning_rate = lr;
}

std::unique_ptr<Matrix> FullyConnected::predict(std::unique_ptr<Matrix>& inputs) {
	std::unique_ptr<Matrix> multiplied = multiply(FullyConnected::weights, inputs);
	std::unique_ptr<Matrix> out = add(multiplied, FullyConnected::bias);

	FullyConnected::hidden_layer = inputs->clone();

	return out;
}

std::unique_ptr<Matrix> FullyConnected::train(std::unique_ptr<Matrix>& errors) {
	std::unique_ptr<Matrix> inputs_transposed = FullyConnected::hidden_layer->transpose();
	std::unique_ptr<Matrix> weight_adjustments = multiply(errors, inputs_transposed);

	std::unique_ptr<Matrix> weights_transposed = FullyConnected::weights->transpose();
	std::unique_ptr<Matrix> back_errors_raw = multiply(weights_transposed, errors);

	std::unique_ptr<Matrix> weight_adjustments_lr = multiplyScalar(weight_adjustments, FullyConnected::learning_rate);
	std::unique_ptr<Matrix> bias_adjustments_lr = multiplyScalar(errors, FullyConnected::learning_rate);

	FullyConnected::weights = subtract(FullyConnected::weights, weight_adjustments_lr);
	FullyConnected::bias = subtract(FullyConnected::bias, bias_adjustments_lr);

	return back_errors_raw;
}

// This function should only ever be used for deepQlearning where two models are required
void FullyConnected::setNetwork(std::unique_ptr<Matrix>& new_weights, std::unique_ptr<Matrix>& new_bias) {
	FullyConnected::weights = new_weights->clone();
	FullyConnected::bias = new_bias->clone();
}

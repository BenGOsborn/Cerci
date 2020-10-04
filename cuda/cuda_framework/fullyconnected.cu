#include "fullyconnected.cuh"

std::unique_ptr<Matrix> FullyConnected::predict(std::unique_ptr<Matrix>& inputs) {
	std::unique_ptr<Matrix> multiplied = multiply(FullyConnected::network->weights, inputs);
	std::unique_ptr<Matrix> out = add(multiplied, FullyConnected::network->bias);

	std::unique_ptr<Matrix> out_applied;
	if (FullyConnected::activation_function == 0) {
		out_applied = applySigmoid(out);
	} else if (FullyConnected::activation_function == 1) {
		out_applied = applyLRelu(out);
	} else if (FullyConnected::activation_function == 2) {
		out_applied = applySoftmax(out);
	} else if (FullyConnected::activation_function == 3) {
		out_applied = applyTanh(out);
	}

	return out_applied;
}

std::unique_ptr<Matrix> FullyConnected::train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& predicted, std::unique_ptr<Matrix>& errors_raw) {

}

std::unique_ptr<Network> FullyConnected::returnNetwork() {
	std::unique_ptr<Network> ret_network = std::make_unique<Network>(FullyConnected::network->weights, FullyConnected::network->bias);
	return ret_network;
}

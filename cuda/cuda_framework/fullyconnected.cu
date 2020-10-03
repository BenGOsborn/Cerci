#include "fullyconnected.cuh"

FullyConnected::FullyConnected(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, std::function<float()> function) {
	FullyConnected::weights = weight_set->clone();
	FullyConnected::bias = bias_set->clone();
	FullyConnected::activation_func = function;
}

std::unique_ptr<Matrix> FullyConnected::predict(std::unique_ptr<Matrix>& inputs) {
	std::unique_ptr<Matrix> multiplied = multiply(FullyConnected::weights, inputs);
	std::unique_ptr<Matrix> out = add(multiplied, FullyConnected::bias);

	std::unique_ptr<Matrix> outCpy = out->clone();
	// Possibly want to set the lambda function up here with the sum and stuff
	// Throw a spare param thing in incase activation functions need it

	// If I pass throught the lambda function as a functional function it wont be able to be parsed to the GPU
	// Should I make my GPU functions with __device__ or do I need them as __global__ ?

	// If I get rid of the functional part can I just define them as function pointers and then assign it that way?
	std::unique_ptr<Matrix> outApplied = out->apply();
}

std::unique_ptr<Matrix> FullyConnected::train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& predicted, std::unique_ptr<Matrix>& errors_raw) {

}

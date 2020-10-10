#include "optimizers.cuh"
#include "lossfunctions.cuh"

int main() {
	// Init the training values
	int size = 3;
	std::unique_ptr<int[]> spe = std::make_unique<int[]>(2);
	spe[0] = size;
	spe[1] = 1;
	std::unique_ptr<float[]> vals = std::make_unique<float[]>(size);
	vals[0] = 1;
	vals[1] = 0;
	vals[2] = 1;
	std::unique_ptr<Matrix> inputs = std::make_unique<Matrix>(vals, spe);

	std::unique_ptr<Matrix> weights = genRand(size, size);
	std::unique_ptr<Matrix> bias = genRand(size, 1);

	// Layers
	std::unique_ptr<FullyConnected> layer1 = std::make_unique<FullyConnected>(weights, bias);
	std::unique_ptr<Sigmoid> sigmoid1 = std::make_unique<Sigmoid>();

	// Now after we add state to the activation functions we want to add in our loss function and try to back propagate

	// This is where we use the network
	std::unique_ptr<Matrix> prediction = layer1->predict(inputs);
	std::unique_ptr<Matrix> sigmoid_applied = sigmoid1->forwardProp(prediction);

	// This calculates and updates the loss for the network
	std::unique_ptr<Matrix> first_error = mse(sigmoid_applied, inputs);
	std::unique_ptr<Matrix> sigmoid_errors = sigmoid1->backProp(first_error);

	// Problem with this function
	std::unique_ptr<Matrix> update_here = layer1->train(sigmoid_errors);

	return 0;
}
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

	// This is where we use the network
	std::unique_ptr<Matrix> prediction;
	std::unique_ptr<Matrix> sigmoid_applied;
	std::unique_ptr<Matrix> first_error;
	std::unique_ptr<Matrix> sigmoid_errors;
	std::unique_ptr<Matrix> update_here;
	for (int i = 0; i < 100; i++) {
		prediction = layer1->predict(inputs);
		sigmoid_applied = sigmoid1->forwardProp(prediction);

		first_error = mse(sigmoid_applied, inputs);
		sigmoid_errors = sigmoid1->backProp(first_error);
		update_here = layer1->train(sigmoid_errors);
	}
	sigmoid_applied->print();

	// How am I going to make multiple convolutional layers and use the same masks and stuff using my matrix library?
	// I could probably make each into its own sort of layer and then just pass it through as needed
	// Or we could just extend the matrix class into its own tensor class and do all the operations there as a tensor instead?
	// If I was very keen I could turn everything into its own tensor class and then just modify it so that does everything in three dimensions instead of the two (Probably the best bet to keep everything normal)

	return 0;
}
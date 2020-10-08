#include "optimizers.cuh"

std::unique_ptr<Matrix> applyMomentum(std::unique_ptr<Matrix>& p_adjustments, float beta1, std::unique_ptr<Matrix>& current_adjustments) {
	std::unique_ptr<Matrix> first_half = multiplyScalar(p_adjustments, beta1);
	std::unique_ptr<Matrix> second_half = multiplyScalar(current_adjustments, (1-beta1));
	std::unique_ptr<Matrix> p = add(first_half, second_half);

	return p;
}

std::unique_ptr<Matrix> applyRMS(std::unique_ptr<Matrix>& p_adjustments, float beta1, std::unique_ptr<Matrix>& current_adjustments) {
	std::unique_ptr<Matrix> first_half = multiplyScalar(p_adjustments, beta1);

	auto square = [] __host__ __device__(float x) { return x * x; };
	std::unique_ptr<Matrix> squared_adjustments = apply(current_adjustments, square);

	std::unique_ptr<Matrix> second_half = multiplyScalar(squared_adjustments, (1-beta1));
	std::unique_ptr<Matrix> rms = add(first_half, second_half);

	return rms;
}

std::unique_ptr<Matrix> applyCorrection(std::unique_ptr<Matrix>& param, float beta, int iteration) {
	float divide_param = 1 - pow(beta, iteration);
	std::unique_ptr<Matrix> corrected = divideScalar(param, divide_param);
}

// Make sure that this calls the constructor of the previous class too
FullyConnectedAdam::FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 10e-8) : FullyConnected(weight_set, bias_set, lr) {
	std::unique_ptr<int[]> weight_shape = weight_set->returnShape();
	std::unique_ptr<int[]> bias_shape = bias_set->returnShape();

	FullyConnectedAdam::p_weights = genZeros(weight_shape[0], weight_shape[1]);
	FullyConnectedAdam::rms_weights = genZeros(weight_shape[0], weight_shape[1]);
	FullyConnectedAdam::p_bias = genZeros(bias_shape[0], bias_shape[1]);
	FullyConnectedAdam::rms_bias = genZeros(bias_shape[0], bias_shape[1]);
	FullyConnectedAdam::iteration = 0;
}

std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors) {
	// So here we have to implement the whole algorithm from scratch again
}

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
FullyConnectedAdam::FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1, float b1 = 0.9, float b2 = 0.999, float eps = 10e-8) : FullyConnected(weight_set, bias_set, lr) {
	std::unique_ptr<int[]> weight_shape = weight_set->returnShape();
	std::unique_ptr<int[]> bias_shape = bias_set->returnShape();

	FullyConnectedAdam::p_weights = genZeros(weight_shape[0], weight_shape[1]);
	FullyConnectedAdam::rms_weights = genZeros(weight_shape[0], weight_shape[1]);
	FullyConnectedAdam::p_bias = genZeros(bias_shape[0], bias_shape[1]);
	FullyConnectedAdam::rms_bias = genZeros(bias_shape[0], bias_shape[1]);

	FullyConnectedAdam::beta1 = b1;
	FullyConnectedAdam::beta2 = b2;
	FullyConnectedAdam::epsilon = eps;

	FullyConnectedAdam::iteration = 0;
}

std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors) {
	FullyConnectedAdam::iteration += 1;

	std::unique_ptr<Matrix> inputs_transposed = inputs->transpose();
	std::unique_ptr<Matrix> weight_adjustments = multiply(errors, inputs_transposed);

	// Am I going to be able to access the protected weights and biases?
	std::unique_ptr<Matrix> weights_transposed = FullyConnectedAdam::weights->transpose();
	std::unique_ptr<Matrix> back_errors_raw = multiply(errors, weights_transposed);

	// Here is where we implement adam
	// We probably want to do some sort of error checking here before setting the new weights as this value
	FullyConnectedAdam::p_weights = applyMomentum(FullyConnectedAdam::p_weights, FullyConnectedAdam::beta1, weight_adjustments);
	FullyConnectedAdam::p_bias = applyMomentum(FullyConnectedAdam::p_bias, FullyConnectedAdam::beta1, errors);
	FullyConnected::rms_weights = applyRMS(FullyConnectedAdam::rms_weights, FullyConnectedAdam::beta2, weight_adjustments);
	FullyConnected::rms_bias = applyRMS(FullyConnectedAdam::rms_bias, FullyConnectedAdam::beta2, errors);

	std::unique_ptr<Matrix> p_weights_corrected = applyCorrection(FullyConnectedAdam::p_weights, FullyConnectedAdam::beta1, FullyConnectedAdam::iteration);
	std::unique_ptr<Matrix> p_bias_corrected = applyCorrection(FullyConnectedAdam::p_bias, FullyConnectedAdam::beta1, FullyConnectedAdam::iteration);
	std::unique_ptr<Matrix> rms_weights_corrected = applyCorrection(FullyConnectedAdam::rms_weights, FullyConnectedAdam::beta2, FullyConnectedAdam::iteration);
	std::unique_ptr<Matrix> rms_bias_corrected = applyCorrection(FullyConnectedAdam::rms_weights, FullyConnectedAdam::beta2, FullyConnectedAdam::iteration);

	auto denom_function = [=] __host__ __device__(float x) { return sqrt(x) + FullyConnectedAdam::epsilon; };
	std::unique_ptr<Matrix> weights_adam_denom = apply(rms_weights_corrected, denom_function);
	std::unique_ptr<Matrix> bias_adam_denom = apply(rms_bias_corrected, denom_function);

	std::unique_ptr<Matrix> adam_weight_adjustments = divideElementwise(p_weights_corrected, weights_adam_denom);
	std::unique_ptr<Matrix> adam_bias_adjustments = divideElementwise(p_bias_corrected, bias_adam_denom);

	std::unique_ptr<Matrix> weight_adjustments_lr = multiplyScalar(weight_adjustments, FullyConnectedAdam::learning_rate);
	std::unique_ptr<Matrix> bias_adjustments_lr = multiplyScalar(errors, FullyConnectedAdam::learning_rate);
	// End of the adam application

	FullyConnectedAdam::weights = subtract(FullyConnectedAdam::weights, weight_adjustments_lr);
	FullyConnectedAdam::bias = subtract(FullyConnectedAdam::bias, bias_adjustments_lr);

	return back_errors_raw;
}

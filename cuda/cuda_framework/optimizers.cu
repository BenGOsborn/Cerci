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

	return corrected;
}

AdamReturn::AdamReturn(std::unique_ptr<Matrix>& p_new, std::unique_ptr<Matrix>& rms_new, std::unique_ptr<Matrix>& out_adam) {
	AdamReturn::p = p_new->clone();
	AdamReturn::rms = rms_new->clone();
	AdamReturn::adam = out_adam->clone();
}

std::unique_ptr<AdamReturn> applyAdam(std::unique_ptr<Matrix>& prev_p, std::unique_ptr<Matrix>& prev_rms, std::unique_ptr<Matrix>& gradients, float beta1, float beta2, float epsilon, int iteration, float learning_rate) {
	std::unique_ptr<Matrix> applied_p = applyMomentum(prev_p, beta1, gradients);
	std::unique_ptr<Matrix> applied_rms = applyRMS(prev_rms, beta2, gradients);
	std::unique_ptr<Matrix> corrected_p = applyCorrection(applied_p, beta1, iteration);
	std::unique_ptr<Matrix> corrected_rms = applyCorrection(applied_rms, beta2, iteration);

	auto denom_function = [=] __host__ __device__(float x) { return sqrt(x) + epsilon; };
	std::unique_ptr<Matrix> denom = apply(corrected_rms, denom_function);

	std::unique_ptr<Matrix> adam_updates = divideElementwise(corrected_p, denom);
	std::unique_ptr<Matrix> adam_lr = multiplyScalar(adam_updates, learning_rate);

	std::unique_ptr<AdamReturn> ret = std::make_unique<AdamReturn>(applied_p, applied_rms, adam_lr);
	return ret;
}

// Make sure that this calls the constructor of the previous class too
FullyConnectedAdam::FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr, float b1, float b2, float eps) : FullyConnected(weight_set, bias_set, lr) {
	std::unique_ptr<int[]> weight_shape = weight_set->returnShape();
	std::unique_ptr<int[]> bias_shape = bias_set->returnShape();

	FullyConnectedAdam::p_weights = genInit(weight_shape[0], weight_shape[1], 0);
	FullyConnectedAdam::rms_weights = genInit(weight_shape[0], weight_shape[1], 0);
	FullyConnectedAdam::p_bias = genInit(bias_shape[0], bias_shape[1], 0);
	FullyConnectedAdam::rms_bias = genInit(bias_shape[0], bias_shape[1], 0);

	FullyConnectedAdam::beta1 = b1;
	FullyConnectedAdam::beta2 = b2;
	FullyConnectedAdam::epsilon = eps;

	FullyConnectedAdam::iteration = 0;
}

std::unique_ptr<Matrix> FullyConnectedAdam::train(std::unique_ptr<Matrix>& errors) {
	FullyConnectedAdam::iteration += 1;

	std::unique_ptr<Matrix> inputs_transposed = FullyConnectedAdam::hidden_layer->transpose();
	std::unique_ptr<Matrix> weight_adjustments = multiply(errors, inputs_transposed);

	std::unique_ptr<Matrix> weights_transposed = FullyConnectedAdam::weights->transpose();
	std::unique_ptr<Matrix> back_errors_raw = multiply(errors, weights_transposed);

	std::unique_ptr<AdamReturn> weight_rets = applyAdam(FullyConnectedAdam::p_weights, FullyConnectedAdam::rms_weights, weight_adjustments, FullyConnectedAdam::beta1, FullyConnectedAdam::beta2, 
														FullyConnectedAdam::epsilon, FullyConnectedAdam::iteration, FullyConnectedAdam::learning_rate);
	std::unique_ptr<AdamReturn> bias_rets = applyAdam(FullyConnectedAdam::p_bias, FullyConnectedAdam::rms_bias, errors, FullyConnectedAdam::beta1, FullyConnectedAdam::beta2, 
														FullyConnectedAdam::epsilon, FullyConnectedAdam::iteration, FullyConnectedAdam::learning_rate);
	
	FullyConnectedAdam::p_weights = weight_rets->p->clone();
	FullyConnectedAdam::rms_weights = weight_rets->rms->clone();
	FullyConnectedAdam::p_bias = bias_rets->p->clone();
	FullyConnectedAdam::rms_bias = bias_rets->rms->clone();

	FullyConnectedAdam::weights = subtract(FullyConnectedAdam::weights, weight_rets->adam);
	FullyConnectedAdam::bias = subtract(FullyConnectedAdam::bias, bias_rets->adam);

	return back_errors_raw;
}

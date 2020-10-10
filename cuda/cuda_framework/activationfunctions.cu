#include "activationfunctions.cuh"

std::unique_ptr<Matrix> Sigmoid::forwardProp(std::unique_ptr<Matrix>& predictions) {
	auto lambda = [] __host__ __device__(float x) { return 1 / (1 + exp(-1.0*x)); };
	std::unique_ptr<Matrix> applied = apply(predictions, lambda);
	Sigmoid::hidden_layer = predictions->clone();

	return applied;
}
std::unique_ptr<Matrix> Sigmoid::backProp(std::unique_ptr<Matrix>& errors) {
	auto lambda = [] __host__ __device__(float x) { return x*(1-x); };
	std::unique_ptr<Matrix> applied = apply(Sigmoid::hidden_layer, lambda);
	std::unique_ptr<Matrix> combo = multiplyElementwise(applied, errors);

	return combo;
}

std::unique_ptr<Matrix> LRelu::forwardProp(std::unique_ptr<Matrix>& predictions) {
	auto lambda = [] __host__ __device__(float x) { if (x <= 0) { return 0.01*x; } else { return 1.0*x; } };
	std::unique_ptr<Matrix> applied = apply(predictions, lambda);
	LRelu::hidden_layer = predictions->clone();

	return applied;
}
std::unique_ptr<Matrix> LRelu::backProp(std::unique_ptr<Matrix>& errors) {
	auto lambda = [] __host__ __device__(float x) { if (x <= 0) { return 0.01; } else { return 1.0; } };
	std::unique_ptr<Matrix> applied = apply(LRelu::hidden_layer, lambda);
	std::unique_ptr<Matrix> combo = multiplyElementwise(applied, errors);

	return combo;
}

std::unique_ptr<Matrix> Softmax::forwardProp(std::unique_ptr<Matrix>& predictions) {
	auto applyExp = [] __host__ __device__(float x) { return exp(x); };
	std::unique_ptr<Matrix> exp_matrix = apply(predictions, applyExp);
	float sm = sum(exp_matrix);

	auto lambda = [=] __host__ __device__(float x) { return x / sm; };
	std::unique_ptr<Matrix> applied = apply(exp_matrix, lambda);

	Softmax::hidden_layer = predictions->clone();

	return applied;
}
std::unique_ptr<Matrix> Softmax::backProp(std::unique_ptr<Matrix>& errors) {
	auto lambda = [] __host__ __device__(float x) { return x * (1 - x); };
	std::unique_ptr<Matrix> applied = apply(Softmax::hidden_layer, lambda);
	std::unique_ptr<Matrix> combo = multiplyElementwise(applied, errors);

	return combo;
}

std::unique_ptr<Matrix> Tanh::forwardProp(std::unique_ptr<Matrix>& predictions) {
	auto lambda = [] __host__ __device__(float x) { return tanh(x); };
	std::unique_ptr<Matrix> applied = apply(predictions, lambda);
	Tanh::hidden_layer = predictions->clone();
	
	return applied;
}
std::unique_ptr<Matrix> Tanh::backProp(std::unique_ptr<Matrix>& errors) {
	auto lambda = [] __host__ __device__(float x) { return 1 - (x * x); };
	std::unique_ptr<Matrix> applied = apply(Tanh::hidden_layer, lambda);
	std::unique_ptr<Matrix> combo = multiplyElementwise(applied, errors);

	return combo;
}

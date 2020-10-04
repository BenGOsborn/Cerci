#include "activationfunctions.cuh"

std::unique_ptr<Matrix> applySigmoid(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { return x*(1-x); };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}
	auto lambda = [] __host__ __device__(float x) { return exp(x); };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);

	return applied;
}

std::unique_ptr<Matrix> applyLRelu(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { if (x <= 0) { return 0.01; } else { return 1.0; } };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}
	auto lambda = [] __host__ __device__(float x) { if (x <= 0) { return 0.01*x; } else { return 1.0*x; } };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);

	return applied;
}

std::unique_ptr<Matrix> applySoftmax(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { return x * (1 - x); };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}
	auto applyExp = [] __host__ __device__(float x) { return exp(x); };
	std::unique_ptr<Matrix> exp_matrix = apply(matrix, applyExp);
	float sm = sum(exp_matrix);

	auto lambda = [=] __host__ __device__(float x) { return x / sm; };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);

	return applied;
}

std::unique_ptr<Matrix> applyTanh(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { return 1 - (x * x); };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}
	auto lambda = [] __host__ __device__(float x) { return tanh(x); };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);
	
	return applied;
}

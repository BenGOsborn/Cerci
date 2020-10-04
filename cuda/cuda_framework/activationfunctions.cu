#include "activationfunctions.cuh"

std::unique_ptr<Matrix> sigmoid(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { return x*(1-x); };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}
	auto lambda = [] __host__ __device__(float x) { return exp(x); };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);

	return applied;
}

std::unique_ptr<Matrix> lRelu(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { if (x <= 0) { return 0.01; } else { return 1.0; } };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}
	auto lambda = [] __host__ __device__(float x) { if (x <= 0) { return 0.01*x; } else { return 1.0*x; } };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);

	return applied;
}

std::unique_ptr<Matrix> softmax(std::unique_ptr<Matrix>& matrix, bool deriv = false) {
	if (deriv) {
		auto lambda = [] __host__ __device__(float x) { return x * (1 - x); };
		std::unique_ptr<Matrix> applied = apply(matrix, lambda);

		return applied;
	}

	// Maybe I dont need these extra params then...
	auto applyExp = [] __host__ __device__(float x) { return exp(x); };
	std::unique_ptr<Matrix> exp_matrix = apply(matrix, applyExp);
	float sm = sum(exp_matrix);

	auto lambda = [=] __host__ __device__(float x) { return x / sm; };
	std::unique_ptr<Matrix> applied = apply(matrix, lambda);

	return applied;
}

#pragma once
#include "fullyconnected.cuh"
#include <math.h>

class FullyConnectedAdam : public FullyConnected {
protected:
	std::unique_ptr<Matrix> p_weights;
	std::unique_ptr<Matrix> p_bias;
	std::unique_ptr<Matrix> rms_weights;
	std::unique_ptr<Matrix> rms_bias;

	float beta1;
	float beta2;
	float epsilon;

	int iteration;
public:
	FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1, float b1 = 0.9, float b2 = 0.999, float eps = 10e-8);
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors);
};

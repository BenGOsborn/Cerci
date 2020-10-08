#pragma once
#include "fullyconnected.cuh"
#include <math.h>

class FullyConnectedAdam : public FullyConnected {
protected:
	std::unique_ptr<Matrix> p_weights;
	std::unique_ptr<Matrix> p_bias;
	std::unique_ptr<Matrix> rms_weights;
	std::unique_ptr<Matrix> rms_bias;
	int iteration;
public:
	FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1, float beta = 0.9, float beta2 = 0.999, float epsilon = 10e-8);
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors);
};

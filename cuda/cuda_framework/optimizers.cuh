#pragma once
#include "fullyconnected.cuh"
#include <math.h>

class FullyConnectedAdam : public FullyConnected {
protected:
	std::unique_ptr<Matrix> p_weights;
	std::unique_ptr<Matrix> rms_weights;
	std::unique_ptr<Matrix> p_bias;
	std::unique_ptr<Matrix> rms_bias;

	float beta1;
	float beta2;
	float epsilon;

	int iteration;
public:
	FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1, float b1 = 0.9, float b2 = 0.999, float eps = 10e-8);
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& errors);
};

struct AdamReturn {
	std::unique_ptr<Matrix> p;
	std::unique_ptr<Matrix> rms;
	std::unique_ptr<Matrix> adam;
	AdamReturn(std::unique_ptr<Matrix>& p_new, std::unique_ptr<Matrix>& rms_new, std::unique_ptr<Matrix>& out_adam);
};
std::unique_ptr<AdamReturn> applyAdam(std::unique_ptr<Matrix>& prev_p, std::unique_ptr<Matrix>& prev_rms, std::unique_ptr<Matrix>& gradients, float beta1, float beta2, float epsilon, int iteration, float learning_rate);

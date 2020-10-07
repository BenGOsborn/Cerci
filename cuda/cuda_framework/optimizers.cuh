#pragma once
#include "fullyconnected.cuh"

class FullyConnectedAdam : public FullyConnected {
protected:
	std::unique_ptr<Matrix> p_weights;
	std::unique_ptr<Matrix> p_bias;
	std::unique_ptr<Matrix> rms_weights;
	std::unique_ptr<Matrix> rms_bias;
	int iteration;
public:
	std::unique_ptr<Matrix> train(std::unique_ptr<Matrix>& inputs, std::unique_ptr<Matrix>& errors);
};

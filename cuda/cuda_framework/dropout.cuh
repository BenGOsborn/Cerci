#pragma once
#include "matrixfunctions.cuh"

class Dropout {
protected:
	float dropout_rate;
public:
	Dropout(float rate = 0.5);
	std::unique_ptr<Matrix> applyDropout(std::unique_ptr<Matrix>& predictions);
	std::unique_ptr<Matrix> backErrors(std::unique_ptr<Matrix>& errors);
};
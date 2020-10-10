#include "dropout.cuh"

Dropout::Dropout(float rate) {
	if ((rate < 0) || (rate >= 1)) throw std::invalid_argument("Rate must be >= 0 and < 1!");
	Dropout::dropout_rate = rate;
}

std::unique_ptr<Matrix> Dropout::applyDropout(std::unique_ptr<Matrix>& predictions) {
	// So first we want to create a mask with the same shape as in the input 
	int size = predictions->returnSize();
	std::unique_ptr<int[]> pred_shape = predictions->returnShape();
	std::unique_ptr<float[]> vals = std::make_unique<float[]>(size);
	for (int i = 0; i < size; i++) {
		if (1.0 * (rand() % 1000) / 1000 < Dropout::dropout_rate) {
			vals[i] = 1.0 / (1 - Dropout::dropout_rate);
		}
		else {
			vals[i] = 0.0f;
		}
	}
	std::unique_ptr<Matrix> dropout_mask = std::make_unique<Matrix>(vals, pred_shape);
	// Is it needed to have all of these clones here?
	Dropout::mask = dropout_mask->clone();
	std::unique_ptr<Matrix> mask_applied = multiplyElementwise(predictions, dropout_mask);

	return mask_applied;
}

std::unique_ptr<Matrix> Dropout::backErrors(std::unique_ptr<Matrix>& errors) {
	std::unique_ptr<Matrix> applied = multiplyElementwise(errors, Dropout::mask);
	
	return applied;
}

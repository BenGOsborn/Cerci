#pragma once
#include <iostream>

class Tensor {
protected:
	std::unique_ptr<float[]> tensor;
	std::unique_ptr<int[]> shape; // 0 is the depth, 1 is the rows, 2 is the cols
	int size_matrix;
	int size_tensor;
public:
	Tensor(std::unique_ptr<float[]>& in_tensor, std::unique_ptr<int[]>& in_shape); 
	void print();
	std::unique_ptr<Tensor> clone();
	std::unique_ptr<float[]> returnTensor();
	std::unique_ptr<int[]> returnShape();
	int returnTensorSize();
	int returnMatrixSize();
};
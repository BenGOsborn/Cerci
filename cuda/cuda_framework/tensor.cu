#include "tensor.cuh"

// Maybe theres a better way instead of doing memcpy
Tensor::Tensor(std::unique_ptr<float[]>& in_tensor, std::unique_ptr<int[]>& in_shape) {
	Tensor::size_matrix = in_shape[1] * in_shape[2];
	Tensor::size_tensor = in_shape[0] * in_shape[1] * in_shape[2];

	Tensor::shape = std::make_unique<int[]>(3);
	memcpy(Tensor::shape.get(), in_shape.get(), 3 * sizeof(int));

	Tensor::tensor = std::make_unique<float[]>(Tensor::size_tensor);
	memcpy(Tensor::tensor.get(), in_tensor.get(), Tensor::size_tensor * sizeof(float));
}

void Tensor::print() {
	for (int depth = 0; depth < Tensor::shape[0]; depth++) {
		for (int row = 0; row < Tensor::shape[1]; row++) {
			for (int col = 0; col < Tensor::shape[2]; col++) {
				std::cout << tensor[depth * Tensor::size_matrix + row * Tensor::shape[2] + col] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

std::unique_ptr<Tensor> Tensor::clone() {
	std::unique_ptr<Tensor> ret_tensor(new Tensor(Tensor::tensor, Tensor::shape));

	return ret_tensor;
}

std::unique_ptr<float[]> Tensor::returnTensor() {
	std::unique_ptr<float[]> ret_tensor(new float[Tensor::size_tensor]);
	memcpy(ret_tensor.get(), Tensor::tensor.get(), Tensor::size_tensor * sizeof(float));

	return ret_tensor;
}

std::unique_ptr<int[]> Tensor::returnShape() {
	std::unique_ptr<int[]> ret_shape(new int[3]);
	memcpy(ret_shape.get(), Tensor::shape.get(), 3 * sizeof(float));

	return ret_shape;
}

int Tensor::returnMatrixSize() {
	return Tensor::size_matrix;
}

int Tensor::returnTensorSize() {
	return Tensor::size_tensor;
}

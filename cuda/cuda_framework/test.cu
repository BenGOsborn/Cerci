#include "matrix.cuh"

int main() {
	std::unique_ptr<Matrix> mat = genRand(3, 4);
	std::unique_ptr<Matrix> transposed = mat->transpose();
	transposed->print();
}
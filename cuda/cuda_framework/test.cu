#include "matrix.cuh"
#include "matrixfunctions.cuh"

int main() {
	std::unique_ptr<Matrix> mat1 = genRand(3, 4);

	auto plus = [] __host__ __device__ (float x) { return x*10; };
	std::unique_ptr<Matrix> applied = apply(mat1, plus);
	applied->print();

	return 0;
}
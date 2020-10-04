#include "matrix.cuh"
// #include "fullyconnected.cuh"

int main() {
	std::unique_ptr<Matrix> mat = genRand(3, 4);
	// Could probably take in the argument as a normal function and then parse it through as the gpu function
	
	auto plus = [] (float x) { return x*10; };
	std::unique_ptr<Matrix> funcApplied = mat->apply(plus);
	funcApplied->print();
}
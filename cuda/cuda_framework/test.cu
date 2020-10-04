#include "matrix.cuh"
// #include "fullyconnected.cuh"

int main() {
	std::unique_ptr<Matrix> mat = genRand(3, 4);
	// Could probably take in the argument as a normal function and then parse it through as the gpu function
	
	// I wonder if it too gets parsed as a function pointer...
	auto plus = [] (float x) { return x + 50; };
	std::unique_ptr<Matrix> funcApplied = mat->apply(plus);
	funcApplied->print();
}
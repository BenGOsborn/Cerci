#include "tensoroperations.cuh"

#include <iostream>

int main() {
    int size = 5;
    std::unique_ptr<float[]> mat1(new float[size]{1, 1, 1, 1, 1});
    std::unique_ptr<float[]> mat2(new float[size]{2, 2, 2, 2, 2});

    std::unique_ptr<float[]> mat3 = CUDAaddElementwise(mat1, mat2, size);

    for (int i = 0; i < size; i++) {
        std::cout << mat3[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
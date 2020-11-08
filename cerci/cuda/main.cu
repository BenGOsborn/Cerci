#include "tensoroperations.cuh"

#include <iostream>
#include <stdlib.h>

int main() {

    int size = 1 << 16;
    int dim_size = 3;
    std::unique_ptr<int[]> dims(new int[dim_size]{64, 64, 16});
    std::unique_ptr<float[]> mat(new float[size]);
    for (int i = 0; i < size; i++) {
        mat[i] = rand() % 100;
    }

    // There is a problem with my CUDAtranspose function
    std::unique_ptr<float[]> out_mat = CUDAtranspose(mat, dims, dim_size, size);

    for (int i = 0; i < size; i++) {
        std::cout << out_mat[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
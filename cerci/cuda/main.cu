#include "tensoroperations.cuh"

#include <iostream>
#include <stdlib.h>

int main() {

    int size1 = 1 << 3;
    int dim_size1 = 3;
    std::unique_ptr<int[]> dims1(new int[dim_size1]{2, 2, 2});
    std::unique_ptr<float[]> mat1(new float[size1]);
    for (int i = 0; i < size1; i++) {
        mat1[i] = rand() % 100;
    }

    int size2 = 1 << 3;
    int dim_size2 = 3;
    std::unique_ptr<int[]> dims2(new int[dim_size2]{2, 2, 2});
    std::unique_ptr<float[]> mat2(new float[size2]);
    for (int i = 0; i < size2; i++) {
        mat2[i] = rand() % 100;
    }

    // It is doing it for the wrong values
    std::unique_ptr<float[]> out_mat = CUDAmultiply(mat1, dims1, dim_size1, size1, mat2, dims2, dim_size2, size2);

    for (int i = 0; i < dims1[0] * dims1[1] * dims2[0]; i++) {
        std::cout << out_mat[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
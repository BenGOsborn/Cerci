#include "tensoroperations.cuh"

#include <iostream>
#include <stdlib.h>

int main() {

    int size1 = 18;
    int dim_size1 = 3;
    std::unique_ptr<int[]> dims1(new int[dim_size1]{3, 3, 2});
    std::unique_ptr<float[]> mat1(new float[size1]);
    for (int i = 0; i < size1; i++) {
        mat1[i] = 2;
    }

    // Nothing is being returned
    std::unique_ptr<float[]> out_mat = CUDApad(mat1, dims1, dim_size1, size1, 0, 0, 0, 0, 1, 1);

    for (int i = 0; i < 98; i++) {
        std::cout << out_mat[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
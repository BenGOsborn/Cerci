#include "tensoroperations.cuh"

#include <iostream>
#include <stdlib.h>

int main() {

    int size1 = 1 << 5;
    int dim_size1 = 3;
    std::unique_ptr<int[]> dims1(new int[dim_size1]{4, 4, 2});
    std::unique_ptr<float[]> mat1(new float[size1]);
    for (int i = 0; i < size1; i++) {
        mat1[i] = 2;
    }

    // So now it is only doing the pooling for the first layer and not any of the others for som reason?
    std::unique_ptr<float[]> out_mat = CUDApad(mat1, dims1, dim_size1, size1, 2, 2, 2, 2, 2, 2);

    for (int i = 0; i < size1 * 100; i++) {
        std::cout << out_mat[i] << " ";
    }

    return 0;
}
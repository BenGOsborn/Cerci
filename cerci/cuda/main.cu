#include "tensoroperations.cuh"

#include <iostream>
#include <stdlib.h>

int main() {

    int size1 = 1 << 10;
    int dim_size1 = 3;
    std::unique_ptr<int[]> dims1(new int[dim_size1]{16, 16, 4});
    std::unique_ptr<float[]> mat1(new float[size1]);
    for (int i = 0; i < size1; i++) {
        mat1[i] = 3;
    }

    // So now it is only doing the pooling for the first layer and not any of the others for som reason?
    std::unique_ptr<float[]> out_mat = CUDAmaxPooling(mat1, dims1, dim_size1, size1, 2, 2, 2, 2);

    for (int i = 0; i < size1; i++) {
        std::cout << out_mat[i] << " ";
    }
    std::cout << "\nVS UNPOOLED:" << std::endl;

    for (int i = 0; i < size1; i++) {
        std::cout << mat1[i] << " ";
    }

    return 0;
}
#include "tensoroperations.cuh"

#include <iostream>

int main() {
    int size = 6;
    int dim_size = 2;
    std::unique_ptr<int[]> dims(new int[dim_size]{3, 2});
    std::unique_ptr<float[]> mat(new float[size]{1, 2, 3, 4, 5, 6});

    // There is a problem with my CUDAtranspose function
    std::unique_ptr<float[]> out_mat = CUDAtranspose(mat, dims, dim_size, size);

    for (int i = 0; i < size; i++) {
        std::cout << out_mat[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
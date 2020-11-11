#include "tensoroperations.cuh"

#include <iostream>
#include <stdlib.h>

int main() {

    int size1 = 4;
    int dim_size1 = 5;
    std::unique_ptr<int[]> dims1(new int[dim_size1]{1, 1, 1, 2, 2});
    std::unique_ptr<float[]> mat1(new float[size1]{3, 2, 1, 0});

    int scale_factor = 2;

    // I should check all of the functions working in higher dimensions
    // This should output 3 3 2 2 1 1 0 0 (refer to the diagram)
    // Maybe we would be better off to do the python version and make the tensor so it can be visualized what is happening?
    std::unique_ptr<float[]> out_mat = CUDAdupe(mat1, dims1, dim_size1, size1, scale_factor); // This isnt doing the full dupe only returns zeros

    for (int i = 0; i < size1 * scale_factor; i++) {
        std::cout << out_mat[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
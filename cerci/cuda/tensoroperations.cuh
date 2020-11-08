#pragma once

// Perform the error checking on the python higher level part

#include <memory>
#include <cmath>

// Perform cleanup and standardization of different function

std::unique_ptr<float[]> CUDAaddElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAmultiplyElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAdivideElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAsubtractElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDApowerElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);

std::unique_ptr<float[]> CUDAtranspose(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size);
std::unique_ptr<float[]> CUDAmultiply(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims);
// I need a kernel function and a reverse kernel function with strides
// I need a pooling function with strides
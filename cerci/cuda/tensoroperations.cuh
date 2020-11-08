#pragma once

// Perform the error checking on the python higher level part

#include <memory>
#include <cmath>

// Elementwise operations
std::unique_ptr<float[]> CUDAaddElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAmultiplyElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAdivideElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAsubtractElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDApowerElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);

// Tricky operations

// In theory we can think of infinite dimensions in terms of just a very long third dimension of different sections
std::unique_ptr<float[]> CUDAtranspose(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size);
std::unique_ptr<float[]> CUDAmultiply(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims);

// Apply function down here? Is it necessary? I could just use the tensors into the function and then have the operations applied
// Use the function and the template I had used before
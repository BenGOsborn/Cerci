#pragma once

// Perform the error checking on the python higher level part
// Maybe we want to change the block sizes so that there are more threads for different GPU's

#include <iostream>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cout << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

#include <memory>
#include <cmath>

// Perform cleanup and standardization of different function
// Organize functions into classes

std::unique_ptr<float[]> CUDAaddElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAmultiplyElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAdivideElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDAsubtractElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);
std::unique_ptr<float[]> CUDApowerElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size);

std::unique_ptr<float[]> CUDAtranspose(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size);
std::unique_ptr<float[]> CUDAmultiply(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims, int in_ptr2_dims_size, int ptr2_size);

// We want to cope with multiple convolutional tensors over multiple dimensions for parallelism
// I need a dilation function which can handle any sort of dilation
// I need a pooling and a pooling reverse
std::unique_ptr<float[]> CUDAmaxPooling(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, int kernel_cols, int kernel_rows, int stride_cols, int stride_rows);

// How do I reverse the padding layer ? 
std::unique_ptr<float[]> CUDApad(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, int pad_left, int pad_right, int pad_up, int pad_down, int pad_between_cols, int pad_between_rows);
#include "tensoroperations.cuh"

__global__
void addD(int size, int& ptr1, int& ptr2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}

std::unique_ptr<float[]> CUDAaddElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int>& in_ptr1_dims, int dim_size1, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int>& in_ptr2_dims, int dim_size2) {
    // We could just assume that they are of the same length...?
    // Its cheaper to do the error checking here than to do it on init of each tensor

    if (dim_size1 != dim_size2) throw std::invalid_argument("Pointers are of different sizes!");

    // I could do a try statement here which checks outside of the lengths of the dimsize to see if it can go any further, then if it can returns failure
    bool valid = false;
    try {
        in_ptr1_dims[dim_size1];
        valid = false;
        break;
        in_ptr2_dims[dim_size1];
        valid = false;
    } catch (const std::exception& e) { 
        // This is a bit broken actually
        throw std::invalid_argument("Sizes do not match the pointer size!");
    }

    int size;
    for (int i = 0; i < dim_size1; i++) {
       if (in_ptr1_dims[i] != in_ptr2_dims[i]) throw std::invalid_argument("Dimensions are of different sizes!"); 
       size *= in_ptr1_dims[i];
    }

    int bytes = size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);
    cudaMalloc(&gpu_ptr3, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), bytes, cudaMemcpyHostToDevice);

    int dimGridX = (size + THREAD_SIZE - 1) / THREAD_SIZE;
    addD <<< dimGradX, THREAD_SIZE >>> (size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unqiue_ptr<float[]> out_ptr3(new float[int_ptr1_size]);
    cudaMemcpy(out_ptr3.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr3;
}
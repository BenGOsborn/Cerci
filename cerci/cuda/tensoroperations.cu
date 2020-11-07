#include "tensoroperations.cuh"

const int THREAD_SIZE = 1 << 10;
const int BLOCK_SIZE = 1 << 5;

__global__
void addElementwiseD(int size, float* ptr1, float* ptr2, float* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr3[idx] = ptr1[idx] + ptr2[idx];
}

std::unique_ptr<float[]> CUDAaddElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size) {
    int bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);
    cudaMalloc(&gpu_ptr3, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE - 1) / THREAD_SIZE;
    addElementwiseD <<< dimGridX, THREAD_SIZE >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
void subtractElementwiseD(int size, float* ptr1, float* ptr2, float* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr3[idx] = ptr1[idx] - ptr2[idx];
}

std::unique_ptr<float[]> CUDAsubtractElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size) {
    int bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);
    cudaMalloc(&gpu_ptr3, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE - 1) / THREAD_SIZE;
    subtractElementwiseD <<< dimGridX, THREAD_SIZE >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
void multiplyElementwiseD(int size, float* ptr1, float* ptr2, float* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr3[idx] = ptr1[idx] * ptr2[idx];
}

std::unique_ptr<float[]> CUDAmultiplyElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size) {
    int bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);
    cudaMalloc(&gpu_ptr3, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE - 1) / THREAD_SIZE;
    multiplyElementwiseD <<< dimGridX, THREAD_SIZE >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
void divideElementwiseD(int size, float* ptr1, float* ptr2, float* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr3[idx] = ptr1[idx] / ptr2[idx];
}

std::unique_ptr<float[]> CUDAdivideElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size) {
    int bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);
    cudaMalloc(&gpu_ptr3, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE - 1) / THREAD_SIZE;
    divideElementwiseD <<< dimGridX, THREAD_SIZE >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
void powerElementwiseD(int size, float* ptr1, float* ptr2, float* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr3[idx] = std::pow(ptr1[idx], ptr2[idx]);
}

std::unique_ptr<float[]> CUDApowerElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size) {
    int bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);
    cudaMalloc(&gpu_ptr3, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE - 1) / THREAD_SIZE;
    powerElementwiseD <<< dimGridX, THREAD_SIZE >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
// I might need a seperate pointer for this because it would of overwritten the value
void transposeD(int rows, int cols, int depths, float* ptr1, float* ptr2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    // Of course this is going to need a Z coordinate for the infinite dimensions it can take
    if ((depth < depths) && (row < rows) && (col < cols)) ptr2[depth * rows * cols + row * cols + col] = ptr1[depth * rows * cols + col * rows + row];
}

// I need to reformat all of the other functions to fit this
std::unique_ptr<float[]> CUDAtranspose(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size) {
    int bytes = ptr1_size * sizeof(float);

    int cols = in_ptr1_dims[0];
    int rows = in_ptr1_dims[1];
    // Is there a faster way to do this
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    float* gpu_ptr1;
    float* gpu_ptr2;
    cudaMalloc(&gpu_ptr1, bytes);
    cudaMalloc(&gpu_ptr2, bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), bytes, cudaMemcpyHostToDevice);

    // It is failing because I am trying to launch too many threads at once
    int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE; // We divide it because this is how many blocks we need to have and then there is BLOCK_SIZE num of threads per block
    int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_depth = (depths + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows, grid_depth);
    dim3 dimThreads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    
    transposeD <<< dimGrid, dimThreads >>> (cols, rows, depths, gpu_ptr1, gpu_ptr2);
    cudaErr( cudaPeekAtLastError() );

    // Change all of these from out_ptr3 to out_ptr as required
    std::unique_ptr<float[]> out_ptr(new float[ptr1_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
   
    return out_ptr;
} 
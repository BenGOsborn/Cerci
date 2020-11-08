#include "tensoroperations.cuh"

const int THREAD_SIZE_XY = 1 << 10;
const int THREAD_SIZE_Z = 1 << 6;

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

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    addElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

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

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    subtractElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

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

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    multiplyElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

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

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    divideElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

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

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    powerElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
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

    int grid_cols = (cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    transposeD <<< gridSize, threadSize >>> (rows, cols, depths, gpu_ptr1, gpu_ptr2);

    std::unique_ptr<float[]> out_ptr(new float[ptr1_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
   
    return out_ptr;
} 

__global__
void multiplyD(int rows, int shared, int cols, int depths, float* ptr1, float* ptr2, float* ptr3) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    float sum;
    if ((col < cols) && (row < rows) && (depth < depths)) {
        sum = 0;
        for (int i = 0; i < shared; i++) {
            sum += ptr1[depth * rows * cols + row * shared + i] * ptr2[depth * rows * cols + i * cols + col];
        }
        ptr3[depth * rows * cols + row * cols + col] = sum;
    }
}

std::unique_ptr<float[]> CUDAmultiply(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims, int in_ptr2_dims_size, int ptr2_size) {
    // This is going to make the assumption that they are equal

    // Maybe move all of the byte size code below the rows and cols allocations

    int ptr1_rows = in_ptr1_dims[1];
    int ptr2_cols = in_ptr2_dims[0];
    int shared_size = in_ptr1_dims[0];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) { // In theory the rest of the dims after should be the exact same if we assume they are correct
        depths *= in_ptr1_dims[i];
    } 

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);
    int gpu_ptr2_bytes = ptr2_size * sizeof(float);
    int gpu_ptr3_bytes = ptr1_rows * ptr2_cols * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr3_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr2_bytes, cudaMemcpyHostToDevice);

    // How is this going to be structured? How many threads do we need for each?
    int grid_cols = (ptr2_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    multiplyD <<< gridSize, threadSize >>> (ptr1_rows, shared_size, ptr2_cols, depths, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr1_rows * ptr2_cols]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr3_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);

    return out_ptr;
}
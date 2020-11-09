#include "tensoroperations.cuh"

const int THREAD_SIZE_XY = 1 << 10;
const int THREAD_SIZE_Z = 1 << 6;

__global__
void addElementwiseD(int size, float* ptr1, float* ptr2, float* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr3[idx] = ptr1[idx] + ptr2[idx];
}

std::unique_ptr<float[]> CUDAaddElementwise(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<float[]>& in_ptr2, int ptr_size) {
    int gpu_ptr1_bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr1_bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    addElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

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
    int gpu_ptr1_bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr1_bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    subtractElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

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
    int gpu_ptr1_bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr1_bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    multiplyElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

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
    int gpu_ptr1_bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr1_bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    divideElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

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
    int gpu_ptr1_bytes = ptr_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr1_bytes);

    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int dimGridX = (ptr_size + THREAD_SIZE_XY - 1) / THREAD_SIZE_XY;
    powerElementwiseD <<< dimGridX, THREAD_SIZE_XY >>> (ptr_size, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);
   
    return out_ptr;
}

__global__
void transposeD(int cols, int rows, int depths, float* ptr1, float* ptr2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    // Of course this is going to need a Z coordinate for the infinite dimensions it can take
    if ((col < cols) && (row < rows) && (depth < depths)) ptr2[depth * rows * cols + row * cols + col] = ptr1[depth * rows * cols + col * rows + row];
}

// I need to reformat all of the other functions to fit this
std::unique_ptr<float[]> CUDAtranspose(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size) {
    int cols = in_ptr1_dims[0];
    int rows = in_ptr1_dims[1];
    // Is there a faster way to do this
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr1_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int grid_cols = (cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    transposeD <<< gridSize, threadSize >>> (cols, rows, depths, gpu_ptr1, gpu_ptr2);

    std::unique_ptr<float[]> out_ptr(new float[ptr1_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
   
    return out_ptr;
} 

__global__
void multiplyD(int cols, int shared, int rows, int depths, float* ptr1, float* ptr2, float* ptr3) {
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
    int ptr1_rows = in_ptr1_dims[1];
    int ptr2_cols = in_ptr2_dims[0];
    int shared_size = in_ptr1_dims[0];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) { // In theory the rest of the dims after should be the exact same if we assume they are correct
        depths *= in_ptr1_dims[i];
    } 

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);
    int gpu_ptr2_bytes = ptr2_size * sizeof(float);
    int gpu_ptr3_bytes = depths * ptr1_rows * ptr2_cols * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr3_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr2_bytes, cudaMemcpyHostToDevice);

    int grid_cols = (ptr2_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    multiplyD <<< gridSize, threadSize >>> (ptr2_cols, shared_size, ptr1_rows, depths, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[depths * ptr1_rows * ptr2_cols]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr3_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);

    return out_ptr;
}

__global__
void maxPoolingD(int cols, int rows, int depths, int kernel_cols, int kernel_rows, int stride_cols, int stride_rows, float* ptr1, float* ptr2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Col of the unpooled ptr
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row of the unpooled ptr
    int depth = blockIdx.z * blockDim.z + threadIdx.z; // Depth of the unpooled ptr

    if ((col < cols - kernel_cols + 1) && (row < rows - kernel_rows + 1) && (depth < depths)) {
        if ((col % stride_cols == 0) && (row % stride_rows == 0)) {

            int max = ptr1[depth * rows * cols + row * cols + col];
            int comparison;

            for (int i = 0; i < kernel_rows; i++) {
                for (int j = 0; j < kernel_cols; j++) {

                    comparison = ptr1[depth * rows * cols + (row + i) * cols + (col + j)];
                    if (max < comparison) max = comparison; 
                    
                }
            }

            int pooled_cols_size = (cols - kernel_cols + stride_cols) / stride_cols;
            int pooled_rows_size = (rows - kernel_rows + stride_rows) / stride_rows;

            int pooled_col = (col - kernel_cols + stride_cols) / stride_cols;
            if (pooled_col < 0) pooled_col = 0;
            int pooled_row = (row - kernel_rows + stride_rows) / stride_rows;
            if (pooled_row < 0) pooled_row = 0;

            ptr2[depth * pooled_rows_size * pooled_cols_size + pooled_row * pooled_cols_size + pooled_col] = max;
        }
    }
}

std::unique_ptr<float[]> CUDAmaxPooling(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, int kernel_cols, int kernel_rows, int stride_cols, int stride_rows) {
    // This is raw dimensions
    int ptr1_cols = in_ptr1_dims[0];
    int ptr1_rows = in_ptr1_dims[1];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    // This is pooled dims
    int ptr2_cols = (ptr1_cols - kernel_cols + stride_cols) / stride_cols;
    int ptr2_rows = (ptr1_rows - kernel_rows + stride_rows) / stride_rows;
    int ptr2_size = ptr2_cols * ptr2_rows * depths;

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);
    int gpu_ptr2_bytes = ptr2_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int grid_cols = (ptr1_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    maxPoolingD <<< gridSize, threadSize >>> (ptr1_cols, ptr1_rows, depths, kernel_rows, kernel_cols, stride_cols, stride_cols, gpu_ptr1, gpu_ptr2);

    std::unique_ptr<float[]> out_ptr(new float[ptr2_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, gpu_ptr2_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    
    return out_ptr;
}

// Maybe for the max poolingD I should do what I did here and include the padded and non padded dimensions and such for easier passing
__global__
void padD(int cols, int rows, int padded_cols, int padded_rows, int depths, int pad_left, int pad_right, int pad_up, int pad_down, int pad_between_cols, int pad_between_rows, float* ptr1, float* ptr2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Col of the unpadded ptr
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row of the unpadded ptr
    int depth = blockIdx.z * blockDim.z + threadIdx.z; // Depth of the unpadded ptr

    // This is the col row and depth for the unpadded dimensions
    if ((col < cols) && (row < rows) && (depth < depths)) {
        ptr2[depth * padded_rows * padded_cols + (pad_up + row * (pad_between_rows + 1)) * padded_cols + (pad_left + col * (pad_between_cols + 1))] = ptr1[depth * rows * cols + row * cols + col];
    }
}

std::unique_ptr<float[]> CUDApad(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, int pad_left, int pad_right, int pad_up, int pad_down, int pad_between_cols, int pad_between_rows) {
    int ptr1_cols = in_ptr1_dims[0];
    int ptr1_rows = in_ptr1_dims[1];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    int ptr2_cols = pad_left + pad_right + ptr1_cols + pad_between_cols * (ptr1_cols - 1);
    int ptr2_rows = pad_up + pad_down + ptr1_rows + pad_between_rows * (ptr1_rows - 1);
    int ptr2_size = ptr2_cols * ptr2_rows * depths; // Need to standardize this across all of the functions

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);
    int gpu_ptr2_bytes = ptr2_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int grid_cols = (ptr1_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    padD <<< gridSize, threadSize >>> (ptr1_cols, ptr1_rows, ptr2_cols, ptr2_rows, depths, pad_left, pad_right, pad_up, pad_down, pad_between_cols, pad_between_rows, gpu_ptr1, gpu_ptr2);

    std::unique_ptr<float[]> out_ptr(new float[ptr2_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, gpu_ptr2_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);

    return out_ptr;
}
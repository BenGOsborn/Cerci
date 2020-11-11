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
void rotateD(int cols, int rows, int depths, float* ptr1, float* ptr2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if ((col < cols) && (row < rows) && (depth < depths)) ptr2[depth * rows * cols + (rows - row - 1) * cols + (cols - col - 1)] = ptr1[depth * rows * cols + row * cols + col];
}

std::unique_ptr<float[]> CUDArotate(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size) {
    int ptr1_cols = in_ptr1_dims[0];
    int ptr1_rows = in_ptr1_dims[1];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    int gpu_ptr_bytes = ptr1_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    cudaMalloc(&gpu_ptr1, gpu_ptr_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr_bytes, cudaMemcpyHostToDevice);

    int grid_cols = (ptr1_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    rotateD <<< gridSize, threadSize >>> (ptr1_cols, ptr1_rows, depths, gpu_ptr1, gpu_ptr2);

    std::unique_ptr<float[]> out_ptr(new float[ptr1_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, gpu_ptr_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);

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

__global__
void poolingDerivD(int cols, int rows, int depths, int kernel_cols, int kernel_rows, int stride_cols, int stride_rows, float* ptr1, float* ptr2, float* ptr3) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Col of the unpooled ptr
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row of the unpooled ptr
    int depth = blockIdx.z * blockDim.z + threadIdx.z; // Depth of the unpooled ptr

    if ((col < cols - kernel_cols + 1) && (row < rows - kernel_rows + 1) && (depth < depths)) {
        if ((col % stride_cols == 0) && (row % stride_rows == 0)) {

            int max = ptr1[depth * rows * cols + row * cols + col];
            int argmax_col = 0;
            int argmax_row = 0;
            int comparison;

            for (int i = 0; i < kernel_rows; i++) {
                for (int j = 0; j < kernel_cols; j++) {

                    comparison = ptr1[depth * rows * cols + (row + i) * cols + (col + j)];
                    if (max < comparison) {
                        max = comparison;
                        argmax_col = j;
                        argmax_row = i;
                    }

                }
            } 

            int pooled_cols_size = (cols - kernel_cols + stride_cols) / stride_cols;
            int pooled_rows_size = (rows - kernel_rows + stride_rows) / stride_rows;

            int pooled_col = (col - kernel_cols + stride_cols) / stride_cols;
            if (pooled_col < 0) pooled_col = 0;
            int pooled_row = (row - kernel_rows + stride_rows) / stride_rows;
            if (pooled_row < 0) pooled_row = 0;

            ptr3[depth * rows * cols + (row + argmax_row) * cols + (col + argmax_col)] += ptr2[depth * pooled_rows_size * pooled_cols_size + pooled_row * pooled_cols_size + pooled_col];

        }
    }
}

std::unique_ptr<float[]> CUDApoolingDeriv(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims, int in_ptr2_dims_size, int ptr2_size,  int kernel_cols, int kernel_rows, int stride_cols, int stride_rows) {
    int ptr1_cols = in_ptr1_dims[0]; // This is the full size unkerneled
    int ptr1_rows = in_ptr1_dims[1];
    int ptr2_cols = in_ptr2_dims[0]; // This is the kernel size!
    int ptr2_rows = in_ptr2_dims[1];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);
    int gpu_ptr2_bytes = ptr2_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    float* gpu_ptr3;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
    cudaMalloc(&gpu_ptr3, gpu_ptr1_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr2, in_ptr2.get(), gpu_ptr2_bytes, cudaMemcpyHostToDevice);

    // Now what memory blocks are we going to use for this?
    int grid_cols = (ptr1_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    poolingDerivD <<< gridSize, threadSize >>> (ptr1_cols, ptr1_rows, depths, kernel_cols, kernel_rows, stride_cols, stride_rows, gpu_ptr1, gpu_ptr2, gpu_ptr3);

    std::unique_ptr<float[]> out_ptr(new float[ptr1_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr1_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);
    cudaFree(gpu_ptr3);

    return out_ptr;
}

__global__
void dupeD(int cols, int rows, int depths, int duped_depths, float* ptr1, float* ptr2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z; // Now represents the depth of the unstreteched size

    if ((col < cols) && (row < rows) && (depth < duped_depths)) {
        int ptr1_depth = depth % depths;
        ptr2[depth * rows * cols + row * cols + col] = ptr1[ptr1_depth * rows * cols + row * cols + col];
    }
}

// This is the broken function
std::unique_ptr<float[]> CUDAdupe(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, int dupe_size) {
    int ptr1_cols = in_ptr1_dims[0];
    int ptr1_rows = in_ptr1_dims[1];
    int depths = 1;
    for (int i = 2; i < in_ptr1_dims_size; i++) {
        depths *= in_ptr1_dims[i];
    }

    int ptr2_depths = dupe_size * depths;
    int ptr2_size = ptr1_cols * ptr1_rows * ptr2_depths;

    int gpu_ptr1_bytes = ptr1_size * sizeof(float);
    int gpu_ptr2_bytes = ptr2_size * sizeof(float);

    float* gpu_ptr1;
    float* gpu_ptr2;
    cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
    cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
    cudaMemcpy(gpu_ptr1, in_ptr1.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);

    int grid_cols = (ptr1_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_rows = (ptr1_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
    int grid_depths = (ptr2_depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z; // This should be the depths of the duped one

    dim3 gridSize(grid_cols, grid_cols, grid_depths);
    dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

    dupeD <<< gridSize, threadSize >>> (ptr1_cols, ptr1_rows, depths, ptr2_depths, gpu_ptr1, gpu_ptr2);

    std::unique_ptr<float[]> out_ptr(new float[ptr2_size]);
    cudaMemcpy(out_ptr.get(), gpu_ptr2, gpu_ptr2_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_ptr1);
    cudaFree(gpu_ptr2);

    return out_ptr;
}

__global__
// What row, col and depth are we choosing? The big one
void convolutionD(int cols, int rows, int kernel_cols, int kernel_rows, int depths, int stride_cols, int stride_rows, float* ptr1, float* ptr2, float* ptr3) {
    // In here we take the correct stride and perform the convolution over that desired block for each element
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if ((col < cols - kernel_cols + 1) && (row < rows - kernel_rows + 1) && (depth < depths)) {

        if ((col % stride_cols == 0) && (row % stride_rows == 0)) {

            float weighted = 0;

            for (int i = 0; i < kernel_rows; i++) {
                for (int j = 0; j < kernel_cols; j++) {

                    weighted += ptr1[depth * rows * cols + (row + i) * cols + (col + j)] * ptr2[depth * rows * cols + i * kernel_cols + j]; // Now I have to do the dot product of the kernel and the convolved

                }
            }

            int weighted_cols_size = (cols - kernel_cols + stride_cols) / stride_cols;
            int weighted_rows_size = (rows - kernel_rows + stride_rows) / stride_rows;

            int weighted_col = (col - kernel_cols + stride_cols) / stride_cols;
            if (weighted_col < 0) weighted_col = 0;
            int weighted_row = (row - kernel_rows + stride_rows) / stride_rows;
            if (weighted_row < 0) weighted_row = 0;            

            ptr3[depth * weighted_rows_size * weighted_cols_size + weighted_row * weighted_cols_size + weighted_col] = weighted;

        }

    }
}

// No bias is required for this
// std::unique_ptr<float[]> CUDAconvolution(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims, int in_ptr2_dims_size, int ptr2_size, int stride_cols, int stride_rows) {

//     // Convolve layer
//     int ptr1_cols = in_ptr1_dims[0];
//     int ptr1_rows = in_ptr1_dims[1];
//     int ptr1_depths = 1;
//     for (int i = 0; i < in_ptr1_dims_size; i++) { 
//         ptr1_depths *= in_ptr1_dims[i];
//     }

//     // Kernel
//     int ptr2_cols = in_ptr2_dims[0];
//     int ptr2_rows = in_ptr2_dims[1];
//     int ptr2_depths = 1;
//     for (int i = 0; i < in_ptr2_dims_size; i++) { 
//         ptr2_depths *= in_ptr2_dims[i];
//     }

//     // This will be the amount to scale the pointers for its depth size
//     int dupe_ptr1 = 1; 
//     if (in_ptr2_dims_size > 3) dupe_ptr1 = in_ptr2_dims[3];
//     int dupe_ptr2 = 1;
//     if (in_ptr1_dims_size > 3) dupe_ptr2 = in_ptr1_dims[3];

//     // We see that the dupe function duplicates every depth in each fourth dimension
//     std::unique_ptr<float[]> ptr1_duped = CUDAdupe(in_ptr1, in_ptr1_dims, in_ptr1_dims_size, ptr1_size, dupe_ptr1); // This will be the ptr1 that has been scaled to match the filter sizes
//     std::unique_ptr<float[]> ptr2_duped = CUDAdupe(in_ptr2, in_ptr2_dims, in_ptr2_dims_size, ptr2_size, dupe_ptr2); // This will scale the kernel to match the amount of input blocks there are

//     int ptr1_duped_size = ptr1_size * dupe_ptr1;
//     int ptr2_duped_size = ptr2_size * dupe_ptr2; // This part could be the problem?

//     // This part is all safe
//     int ptr3_cols = (ptr1_cols - ptr2_cols + stride_cols) / stride_cols;
//     int ptr3_rows = (ptr1_rows - ptr2_rows + stride_rows) / stride_rows;
//     int ptr3_depths = dupe_ptr1 * ptr1_depths;
//     int ptr3_size = ptr3_depths * ptr3_rows * ptr3_cols;

//     int gpu_ptr1_bytes = ptr1_duped_size * sizeof(float); // These must be the wrong allocation sizes
//     int gpu_ptr2_bytes = ptr2_duped_size * sizeof(float);
//     int gpu_ptr3_bytes = ptr3_size * sizeof(float);

//     float* gpu_ptr1; // Convolved
//     float* gpu_ptr2; // Kernel
//     float* gpu_ptr3; // Output
//     cudaMalloc(&gpu_ptr1, gpu_ptr1_bytes);
//     cudaMalloc(&gpu_ptr2, gpu_ptr2_bytes);
//     cudaMalloc(&gpu_ptr3, gpu_ptr3_bytes);
//     cudaMemcpy(gpu_ptr1, ptr1_duped.get(), gpu_ptr1_bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(gpu_ptr2, ptr2_duped.get(), gpu_ptr2_bytes, cudaMemcpyHostToDevice); // The memory allocation for this one is wrong

//     int grid_cols = (ptr3_cols + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
//     int grid_rows = (ptr3_rows + std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z) - 1) / std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z);
//     int grid_depths = (ptr3_depths + THREAD_SIZE_Z - 1) / THREAD_SIZE_Z;

//     dim3 gridSize(grid_cols, grid_cols, grid_depths);
//     dim3 threadSize(std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), std::sqrt(THREAD_SIZE_XY / THREAD_SIZE_Z), THREAD_SIZE_Z);

//     std::unique_ptr<float[]> out_ptr(new float[ptr3_size]);
//     cudaMemcpy(out_ptr.get(), gpu_ptr3, gpu_ptr3_bytes, cudaMemcpyDeviceToHost);

//     cudaFree(gpu_ptr1);
//     cudaFree(gpu_ptr2);
//     cudaFree(gpu_ptr3);

//     return out_ptr;
// }

// std::unique_ptr<float[]> CUDAconvolution(std::unique_ptr<float[]>& in_ptr1, std::unique_ptr<int[]>& in_ptr1_dims, int in_ptr1_dims_size, int ptr1_size, std::unique_ptr<float[]>& in_ptr2, std::unique_ptr<int[]>& in_ptr2_dims, int in_ptr2_dims_size, int ptr2_size, int stride_cols, int stride_rows) {
//     int ptr1_cols = ;
// }

// New Pseudo:
//  Inputs: A layered 4 dimensional input block
//          A layered 4 dimensional weight block (with the same depth as those of the filters)
//          The third dimensions nof each should line up but not the fourth dimension

//  Scaling: For the input block scale them to be the same size as the fourth dimension of the weight block 
//           For the weight blocks, condense it all into a single 3d layer and then scale them by the fourth dimension of the input block

// Post Scaling: Turn the scaled input block into a single three dimensional layer (do this by multiplying the depth by the rest of the size)
//               Turn the scaled weight block into a big single three dimensional block too

// Remaining steps: Perform the convolution across every different subsection
// Output it as a block with dimensions of the new rows and cols, the depth of the original depth of the input block and the fourth dimension of the kernels 

// Post processing ----------- (NOT NEEDED)
//  Do the sum across all of the fourth dimensions into a single third dimension (or something)
//  Add the bias term to each respective element

// Thoughts?
// How would this deal with a block size larger than four dimensions?
// To do so it appears that the dupe function is broken - It does not perform the duplicates properly for just the fourth dimensions, lets check this out
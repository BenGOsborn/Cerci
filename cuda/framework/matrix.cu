#include <stdio.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\nGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void vectorAdd(int n, int *x, int *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + y[i];
}

int main(void) {
    int N = 1<<4;
    int bytes = N*sizeof(int);

    // Allocate memory on the host device
    int *x, *y;
    x = (int*)malloc(bytes);
    y = (int*)malloc(bytes);

    // Allocate mrmoy on the GPU
    int *d_x, *d_y;
    cudaMalloc(&d_x, bytes); 
    cudaMalloc(&d_y, bytes);

    // Initialize values on the host device
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 100;
        y[i] = rand() % 100;
    }

    std::cout << "Initial x: ";
    for (int i = 0; i < N; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << "\nInitial y: ";
    for (int i = 0; i < N; i++) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    // Copy the memory from the host device to the GPU
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements on the GPU
    vectorAdd<<<(N+255)/256, 256>>>(N, d_x, d_y);

    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Final: ";
    for (int i = 0; i < N; i++) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
int main() {
    int N = 1<<2;
    size_t bytes = N*sizeof(int);

    float *x; 
    float *y;

    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
    }

    cudaFree(x);
    cudaFree(y);

    return 0;
}
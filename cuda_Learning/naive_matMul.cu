#include <stdio.h>
#include <cuda_runtime.h>

__global__ void naiveMatMul(float *C, const float *A, const float *B, int m, int n, int k){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < m && col < n){
        float sum = 0.0f;
        for(int l = 0; l < k; l++){
            sum += A[row*k + l] * B[l*n + col];
        }
        C[row*n + col] = sum;
    }
}


int main(){
    const int m = 1024, n = 1024, k = 1024;
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    for(int i = 0; i < m*k; i++) h_A[i] = 1.0f;
    for(int i = 0; i < k*n; i++) h_B[i] = 1.0f;

    float *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, bytes_A);
    cudaMalloc(&dev_B, bytes_B);
    cudaMalloc(&dev_C, bytes_C);

    cudaMemcpy(dev_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n+15)/16, (m+15)/16);
    naiveMatMul<<<blocks, threads>>>(dev_C, dev_A, dev_B, m, n, k);

    cudaMemcpy(h_C, dev_C, bytes_C, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %f (expected %d)\n", h_C[0], k);
    printf("C[1023][1023] = %f (expected %d)\n", h_C[1023*n+1023], k);

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
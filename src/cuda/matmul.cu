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
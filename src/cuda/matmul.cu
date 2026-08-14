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

#define TILE_SIZE 16

__global__ void tiledMatMul(float *C, const float *A, const float *B, int m, int n, int k){
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for(int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++){

        if(row < m && t * TILE_SIZE + threadIdx.x < k)
            tileA[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if(t * TILE_SIZE + threadIdx.y < k && col < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if(row < m && col < n)
        C[row * n + col] = sum;
}
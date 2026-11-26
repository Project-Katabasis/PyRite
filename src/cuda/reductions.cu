#include <stdio.h>
#include <cuda_runtime.h>

__global__ void naiveColSum(float* out, const float* in, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
            sum += in[i * cols + j];
        out[i] = sum;
    }
}

__global__ void naiveSoftmax(float* out, const float* in, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float denom = 0.0f;
        for (int i = 0; i < rows; i++)
            denom += expf(in[i * cols + j]);
        for (int i = 0; i < rows; i++)
            out[i * cols + j] = expf(in[i * cols + j]) / denom;
    }
}
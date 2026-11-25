#pragma once


//elementwise
__global__ void elementwiseAdd(float *c, const float *a, const float *b, int size);
__global__ void scalarMulKernel(float *a, float scalar, int size);
__global__ void elementwiseMulKernel(float *c, float *a, float *b, int size);
__global__ void elementwiseRelu(float *c, float *a, int size);
__global__ void reluBackwardKernel(float *c, float *grad, float *a, int size);
__global__ void transposeKernel(float *c, float *a, int rows, int cols, int size);
__global__ void broadcastKernel(float *c, float *a, int outCols, int size);

//matMul
__global__ void naiveMatMul(float *C, const float *A, const float *B, int m, int n, int k);
__global__ void tiledMatMul(float *C, const float *A, const float *B, int m, int n, int k);

//convolutional operations
__global__ void im2colKernel(
    float* out,          // output: [col_len, n_cols] row-major
    const float* input,  // input: [in_C*in_H*in_W, N] row-major
    int in_C, int in_H, int in_W,
    int kH, int kW,
    int stride, int pad,
    int outH, int outW,
    int N,
    int col_len, int n_cols,
    int out_size
);

__global__ void col2imKernel(
    float* grad_input,   // output: [in_C*in_H*in_W, N]
    const float* cols,   // input: [col_len, n_cols]
    int in_C, int in_H, int in_W,
    int kH, int kW,
    int stride, int pad,
    int outH, int outW,
    int N,
    int col_len, int n_cols,
    int out_size
);


//reductions
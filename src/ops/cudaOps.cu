#include "pyrite/ops/cudaOps.h"
#include "pyrite/cuda/kernels.cuh"
#include <cuda_runtime.h>

static inline int blocks(int n, int tpb = 256) { return (n + tpb - 1) / tpb; }

namespace cuda_ops {

    // ── Matrix ops ────────────────────────────────────────────────────────────
    void matMul(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        naiveMatMul<<<blocks(rows * cols), 256>>>(C, A, B, rows, inner, cols);
        cudaDeviceSynchronize();
    }

    void matAdd(float* C, const float* A, const float* B, int size) {
        elementwiseAdd<<<blocks(size), 256>>>(C, A, B, size);
        cudaDeviceSynchronize();
    }

    // ── Elementwise ops ───────────────────────────────────────────────────────
    void elementMul(float* C, const float* A, const float* B, int size) {
        elementwiseMulKernel<<<blocks(size), 256>>>(C, (float*)A, (float*)B, size);
        cudaDeviceSynchronize();
    }

    void broadcast(float* out, const float* in, int rows, int out_cols) {
        broadcastKernel<<<blocks(rows * out_cols), 256>>>(out, (float*)in, out_cols, rows * out_cols);
        cudaDeviceSynchronize();
    }

    void transpose(float* out, const float* in, int rows, int cols) {
        transposeKernel<<<blocks(rows * cols), 256>>>(out, (float*)in, rows, cols, rows * cols);
        cudaDeviceSynchronize();
    }

    void sqrtElm(float* out, const float* in, int size) {
        sqrtKernel<<<blocks(size), 256>>>(out, in, size);
        cudaDeviceSynchronize();
    }

    // ── Activations ───────────────────────────────────────────────────────────
    void relu(float* out, const float* in, int size) {
        elementwiseRelu<<<blocks(size), 256>>>(out, (float*)in, size);
        cudaDeviceSynchronize();
    }

    void reluBackward(float* out, const float* grad, const float* in, int size) {
        reluBackwardKernel<<<blocks(size), 256>>>(out, (float*)grad, (float*)in, size);
        cudaDeviceSynchronize();
    }

    // ── Reductions ────────────────────────────────────────────────────────────
    void colSum(float* out, const float* in, int rows, int cols) {
        naiveColSum<<<blocks(rows), 256>>>(out, in, rows, cols);
        cudaDeviceSynchronize();
    }

    float sum(const float* in, int size) {
        // reuse naiveColSum treating entire array as 1 row
        float* d_out;
        cudaMalloc(&d_out, sizeof(float));
        naiveColSum<<<1, 1>>>(d_out, in, 1, size);
        cudaDeviceSynchronize();
        float result;
        cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        return result;
    }

    void softmax(float* out, const float* in, int rows, int cols) {
        naiveSoftmax<<<blocks(cols), 256>>>(out, in, rows, cols);
        cudaDeviceSynchronize();
    }

    // ── Convolution ───────────────────────────────────────────────────────────
    void im2col(float* out, const float* input, int in_C, int in_H, int in_W,
                int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols) {
        int col_len  = in_C * kH * kW;
        int out_size = col_len * n_cols;
        im2colKernel<<<blocks(out_size), 256>>>(out, input, in_C, in_H, in_W, kH, kW, stride, pad, outH, outW, N, col_len, n_cols, out_size);
        cudaDeviceSynchronize();
    }

    void col2im(float* grad_input, const float* cols, int in_C, int in_H, int in_W,
                int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols) {
        int col_len  = in_C * kH * kW;
        int out_size = col_len * n_cols;
        col2imKernel<<<blocks(out_size), 256>>>(grad_input, cols, in_C, in_H, in_W, kH, kW, stride, pad, outH, outW, N, col_len, n_cols, out_size);
        cudaDeviceSynchronize();
    }

    void matSub(float* C, const float* A, const float* B, int size) {
    elementwiseSub<<<blocks(size), 256>>>(C, A, B, size);
    cudaDeviceSynchronize();
    }

    void scalarMul(float* out, const float* in, float scalar, int size) {
        scalarMulKernel<<<blocks(size), 256>>>(out, in, scalar, size);
        cudaDeviceSynchronize();
    }

    void scalarAdd(float* out, const float* in, float scalar, int size) {
        scalarAddKernel<<<blocks(size), 256>>>(out, in, scalar, size);
        cudaDeviceSynchronize();
    }

    void elementDiv(float* C, const float* A, const float* B, int size) {
        elementwiseDiv<<<blocks(size), 256>>>(C, A, B, size);
        cudaDeviceSynchronize();
    }


}
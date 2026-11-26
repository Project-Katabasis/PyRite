#pragma once

namespace cuda_ops {

    // ── Matrix ops ────────────────────────────────────────────────────────────
    void matMul(float* C, const float* A, const float* B, int rows, int cols, int inner);
    void matAdd(float* C, const float* A, const float* B, int size);
    void matSub(float* C, const float* A, const float* B, int size);

    // ── Elementwise ops ───────────────────────────────────────────────────────
    void scalarMul(float* out, const float* in, float scalar, int size);
    void scalarAdd(float* out, const float* in, float scalar, int size);
    void elementMul(float* C, const float* A, const float* B, int size);
    void elementDiv(float* C, const float* A, const float* B, int size);
    void sqrtElm(float* out, const float* in, int size);

    // ── Shape ops ─────────────────────────────────────────────────────────────
    void transpose(float* out, const float* in, int rows, int cols);
    void broadcast(float* out, const float* in, int rows, int out_cols);

    // ── Activations ───────────────────────────────────────────────────────────
    void relu(float* out, const float* in, int size);
    void reluBackward(float* out, const float* grad, const float* in, int size);

    // ── Reductions ────────────────────────────────────────────────────────────
    void colSum(float* out, const float* in, int rows, int cols);
    float sum(const float* in, int size);
    void softmax(float* out, const float* in, int rows, int cols);

    // ── Convolution ───────────────────────────────────────────────────────────
    void im2col(float* out, const float* input,
                int in_C, int in_H, int in_W,
                int kH, int kW, int stride, int pad,
                int outH, int outW, int N, int n_cols);

    void col2im(float* grad_input, const float* cols,
                int in_C, int in_H, int in_W,
                int kH, int kW, int stride, int pad,
                int outH, int outW, int N, int n_cols);


}
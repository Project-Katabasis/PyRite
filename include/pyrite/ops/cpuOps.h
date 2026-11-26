#pragma once
#include <cstddef>
#include <vector>
#include <functional>

namespace cpu_ops {

    // ── Matrix operations ─────────────────────────────────────────────────────
    void matMul(float* C, const float* A, const float* B, int rows, int cols, int inner);
    void matAdd(float* C, const float* A, const float* B, int size);
    void matSub(float* C, const float* A, const float* B, int size);

    // ── Elementwise operations ────────────────────────────────────────────────
    void scalarMul(float* out, const float* in, float scalar, int size);
    void scalarAdd(float* out, const float* in, float scalar, int size);
    void elementMul(float* C, const float* A, const float* B, int size);
    void elementDiv(float* C, const float* A, const float* B, int size);

    // ── Shape operations ──────────────────────────────────────────────────────
    void transpose(float* out, const float* in, int rows, int cols);
    void broadcast(float* out, const float* in, int rows, int out_cols);

    // ── Reductions ────────────────────────────────────────────────────────────
    void colSum(float* out, const float* in, int rows, int cols);
    float sum(const float* in, int size);
    void softmax(float* out, const float* in, int rows, int cols);

    // ── Activations ───────────────────────────────────────────────────────────
    void relu(float* out, const float* in, int size);
    void reluBackward(float* out, const float* grad, const float* in, int size);

    // ── Convolution ───────────────────────────────────────────────────────────
    void im2col(float* out, const float* input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols);

    void col2im(float* grad_input, const float* cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols);

    // ── Misc ──────────────────────────────────────────────────────────────────
    void funcApply(float* out, const float* in, int size, const std::function<float(float)>& fn);

    void scatterAdd(float* target, const int* indices, const float* source, int size);
    
    int argmax(const float* in, int size);

    void colSelect(float* out, const float* in, int rows, int in_cols, const int* indices, int n_cols);

    void reorder(float* out, const float* in, const int* dst_to_src, int size);
}
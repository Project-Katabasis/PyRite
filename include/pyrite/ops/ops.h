#pragma once
#include <cstddef>
#include <functional>
#include "pyrite/memory/device.h"

namespace ops {

    // ── Matrix operations ─────────────────────────────────────────────────────
    void matMul(float* C, const float* A, const float* B, int rows, int cols, int inner, Device device);
    void matAdd(float* C, const float* A, const float* B, int size, Device device);
    void matSub(float* C, const float* A, const float* B, int size, Device device);

    // ── Elementwise operations ────────────────────────────────────────────────
    void scalarMul(float* out, const float* in, float scalar, int size, Device device);
    void scalarAdd(float* out, const float* in, float scalar, int size, Device device);
    void elementMul(float* C, const float* A, const float* B, int size, Device device);
    void elementDiv(float* C, const float* A, const float* B, int size, Device device);

    // ── Shape operations ──────────────────────────────────────────────────────
    void transpose(float* out, const float* in, int rows, int cols, Device device);
    void broadcast(float* out, const float* in, int rows, int out_cols, Device device);

    // ── Reductions ────────────────────────────────────────────────────────────
    void colSum(float* out, const float* in, int rows, int cols, Device device);
    float sum(const float* in, int size, Device device);
    void softmax(float* out, const float* in, int rows, int cols, Device device);

    // ── Activations ───────────────────────────────────────────────────────────
    void relu(float* out, const float* in, int size, Device device);
    void reluBackward(float* out, const float* grad, const float* in, int size, Device device);

    // ── Convolution ───────────────────────────────────────────────────────────
    void im2col(float* out, const float* input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols, Device device);

    void col2im(float* grad_input, const float* cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols, Device device);

    // ── Misc ──────────────────────────────────────────────────────────────────
    void scatterAdd(float* target, const int* indices, const float* source, int size, Device device);
    
    int argmax(const float* in, int size, Device device);
    
    void colSelect(float* out, const float* in, int rows, int in_cols, const int* indices, int n_cols, Device device);
    
    void reorder(float* out, const float* in, const int* dst_to_src, int size, Device device);

}
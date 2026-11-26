#include "pyrite/ops/ops.h"
#include "pyrite/ops/cpuOps.h"
// #include "pyrite/ops/cudaOps.cuh"  // uncomment when CUDA ops exist

namespace ops {

    void matMul(float* C, const float* A, const float* B, int rows, int cols, int inner, Device device) {
        if (device == Device::CPU) cpu_ops::matMul(C, A, B, rows, cols, inner);
        // else cuda_ops::matMul(C, A, B, rows, cols, inner);
    }

    void matAdd(float* C, const float* A, const float* B, int size, Device device) {
        if (device == Device::CPU) cpu_ops::matAdd(C, A, B, size);
    }

    void matSub(float* C, const float* A, const float* B, int size, Device device) {
        if (device == Device::CPU) cpu_ops::matSub(C, A, B, size);
    }

    void scalarMul(float* out, const float* in, float scalar, int size, Device device) {
        if (device == Device::CPU) cpu_ops::scalarMul(out, in, scalar, size);
    }

    void scalarAdd(float* out, const float* in, float scalar, int size, Device device) {
        if (device == Device::CPU) cpu_ops::scalarAdd(out, in, scalar, size);
    }

    void elementMul(float* C, const float* A, const float* B, int size, Device device) {
        if (device == Device::CPU) cpu_ops::elementMul(C, A, B, size);
    }

    void elementDiv(float* C, const float* A, const float* B, int size, Device device) {
        if (device == Device::CPU) cpu_ops::elementDiv(C, A, B, size);
    }

    void transpose(float* out, const float* in, int rows, int cols, Device device) {
        if (device == Device::CPU) cpu_ops::transpose(out, in, rows, cols);
    }

    void broadcast(float* out, const float* in, int rows, int out_cols, Device device) {
        if (device == Device::CPU) cpu_ops::broadcast(out, in, rows, out_cols);
    }

    void colSum(float* out, const float* in, int rows, int cols, Device device) {
        if (device == Device::CPU) cpu_ops::colSum(out, in, rows, cols);
    }

    float sum(const float* in, int size, Device device) {
        if (device == Device::CPU) return cpu_ops::sum(in, size);
        return 0.0f;
    }

    void softmax(float* out, const float* in, int rows, int cols, Device device) {
        if (device == Device::CPU) cpu_ops::softmax(out, in, rows, cols);
    }

    void relu(float* out, const float* in, int size, Device device) {
        if (device == Device::CPU) cpu_ops::relu(out, in, size);
    }

    void reluBackward(float* out, const float* grad, const float* in, int size, Device device) {
        if (device == Device::CPU) cpu_ops::reluBackward(out, grad, in, size);
    }

    void im2col(float* out, const float* input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols, Device device) {
        if (device == Device::CPU) cpu_ops::im2col(out, input, in_C, in_H, in_W, kH, kW, stride, pad, outH, outW, N, n_cols);
    }

    void col2im(float* grad_input, const float* cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols, Device device) {
        if (device == Device::CPU) cpu_ops::col2im(grad_input, cols, in_C, in_H, in_W, kH, kW, stride, pad, outH, outW, N, n_cols);
    }

    void scatterAdd(float* target, const int* indices, const float* source, int size, Device device) {
        if (device == Device::CPU) cpu_ops::scatterAdd(target, indices, source, size);
    }

    int argmax(const float* in, int size, Device device) {
        if (device == Device::CPU) return cpu_ops::argmax(in, size);
        return 0;
    }

    void colSelect(float* out, const float* in, int rows, int in_cols, const int* indices, int n_cols, Device device) {
        if (device == Device::CPU) cpu_ops::colSelect(out, in, rows, in_cols, indices, n_cols);
    }

    void reorder(float* out, const float* in, const int* dst_to_src, int size, Device device) {
        if (device == Device::CPU) cpu_ops::reorder(out, in, dst_to_src, size);
    }

}
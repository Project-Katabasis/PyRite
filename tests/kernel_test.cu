/*
 * kernel_test.cu — unit tests for CUDA kernels in PyRite
 *
 * Tests each kernel against the equivalent CPU Tensor:: implementation.
 * A small known input is run through both paths; outputs are compared
 * element-by-element within a float tolerance.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -Iinclude -o tests/kernel_test \
 *        tests/kernel_test.cu src/tensor.cu src/cuda/elementwise.cu \
 *        src/cuda/matmul.cu src/cuda/conv_ops.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>

#include "pyrite/tensor.h"
#include "pyrite/cuda/kernels.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static const float TOL = 1e-5f;

static bool allClose(const std::vector<float>& a,
                     const std::vector<float>& b,
                     float tol = TOL)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            printf("  MISMATCH at index %zu: cpu=%.6f  gpu=%.6f  diff=%.2e\n",
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

static void report(const char* name, bool passed) {
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
}

// Launch config for 1-D kernels
static inline int blocks(int n, int tpb = 256) { return (n + tpb - 1) / tpb; }

// ─────────────────────────────────────────────────────────────────────────────
// im2col
// ─────────────────────────────────────────────────────────────────────────────
/*
 * Input:  1 channel, 4×4 image, batch N=2  → shape [1*4*4, 2] = [16, 2]
 * Kernel: 3×3, stride=1, pad=0
 * outH = outW = 2
 * col_len  = 1*3*3 = 9
 * n_cols   = 2*2*2 = 8
 * out_size = 72
 */
static bool test_im2col() {
    const int in_C=1, in_H=4, in_W=4, kH=3, kW=3, stride=1, pad=0, N=2;
    const int outH=(in_H+2*pad-kH)/stride+1;   // 2
    const int outW=(in_W+2*pad-kW)/stride+1;   // 2
    const int col_len = in_C*kH*kW;            // 9
    const int n_cols  = outH*outW*N;            // 8
    const int out_size = col_len * n_cols;      // 72
    const int in_size  = in_C*in_H*in_W*N;     // 32

    // fill input with 0,1,2,...
    std::vector<float> h_input(in_size);
    for (int i = 0; i < in_size; i++) h_input[i] = (float)i;

    // CPU reference via Tensor::im2col
    Tensor t_input(h_input, {in_C*in_H*in_W, N});
    Tensor cpu_out = Tensor::im2col(t_input, in_C, in_H, in_W, kH, kW, stride, pad);

    // GPU path
    float *d_input, *d_out;
    cudaMalloc(&d_input, in_size  * sizeof(float));
    cudaMalloc(&d_out,   out_size * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);

    im2colKernel<<<blocks(out_size), 256>>>(
        d_out, d_input,
        in_C, in_H, in_W, kH, kW, stride, pad,
        outH, outW, N, col_len, n_cols, out_size);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(out_size);
    cudaMemcpy(gpu_out.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_out);

    return allClose(cpu_out.data, gpu_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// col2im
// ─────────────────────────────────────────────────────────────────────────────
/*
 * Uses the same geometry as im2col test.
 * Feeds a known cols matrix (all ones) through both paths and compares
 * the accumulated gradient tensors.
 */
static bool test_col2im() {
    const int in_C=1, in_H=4, in_W=4, kH=3, kW=3, stride=1, pad=0, N=2;
    const int outH=(in_H+2*pad-kH)/stride+1;
    const int outW=(in_W+2*pad-kW)/stride+1;
    const int col_len  = in_C*kH*kW;
    const int n_cols   = outH*outW*N;
    const int out_size = col_len * n_cols;
    const int grad_size = in_C*in_H*in_W*N;

    // cols input: all ones
    std::vector<float> h_cols(out_size, 1.0f);

    // CPU reference
    Tensor t_cols(h_cols, {col_len, n_cols});
    Tensor cpu_out = Tensor::col2im(t_cols, in_C, in_H, in_W, kH, kW, stride, pad, N);

    // GPU path
    float *d_cols, *d_grad;
    cudaMalloc(&d_cols, out_size  * sizeof(float));
    cudaMalloc(&d_grad, grad_size * sizeof(float));
    cudaMemcpy(d_cols, h_cols.data(), out_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grad, 0, grad_size * sizeof(float));   // must zero before atomicAdd

    col2imKernel<<<blocks(out_size), 256>>>(
        d_grad, d_cols,
        in_C, in_H, in_W, kH, kW, stride, pad,
        outH, outW, N, col_len, n_cols, out_size);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(grad_size);
    cudaMemcpy(gpu_out.data(), d_grad, grad_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_cols); cudaFree(d_grad);

    return allClose(cpu_out.data, gpu_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// transpose
// ─────────────────────────────────────────────────────────────────────────────
static bool test_transpose() {
    const int rows=5, cols=7, size=rows*cols;
    std::vector<float> h_in(size);
    for (int i = 0; i < size; i++) h_in[i] = (float)i;

    Tensor t_in(h_in, {rows, cols});
    Tensor cpu_out = Tensor::transpose(t_in);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    transposeKernel<<<blocks(size), 256>>>(d_out, d_in, rows, cols, size);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(size);
    cudaMemcpy(gpu_out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    return allClose(cpu_out.data, gpu_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast
// ─────────────────────────────────────────────────────────────────────────────
static bool test_broadcast() {
    const int rows=4, out_cols=6, size=rows*out_cols;
    std::vector<float> h_col = {1.0f, 2.0f, 3.0f, 4.0f};  // [4,1]

    Tensor t_col(h_col, {rows, 1});
    Tensor cpu_out = Tensor::broadcast(t_col, out_cols);

    float *d_col, *d_out;
    cudaMalloc(&d_col, rows * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_col, h_col.data(), rows * sizeof(float), cudaMemcpyHostToDevice);

    broadcastKernel<<<blocks(size), 256>>>(d_out, d_col, out_cols, size);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(size);
    cudaMemcpy(gpu_out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_col); cudaFree(d_out);

    return allClose(cpu_out.data, gpu_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// elementwise relu forward
// ─────────────────────────────────────────────────────────────────────────────
static bool test_relu() {
    const int size = 8;
    std::vector<float> h_in = {-2.f, -1.f, 0.f, 1.f, 2.f, -0.5f, 3.f, -3.f};
    std::vector<float> cpu_out(size);
    for (int i = 0; i < size; i++) cpu_out[i] = h_in[i] > 0.f ? h_in[i] : 0.f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    elementwiseRelu<<<blocks(size), 256>>>(d_out, d_in, size);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(size);
    cudaMemcpy(gpu_out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    return allClose(cpu_out, gpu_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== PyRite kernel tests ===\n\n");

    report("im2col",    test_im2col());
    report("col2im",    test_col2im());
    report("transpose", test_transpose());
    report("broadcast", test_broadcast());
    report("relu fwd",  test_relu());

    printf("\nDone.\n");
    return 0;
}

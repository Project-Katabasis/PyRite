#include "pyrite/memory/device.h"
#include "pyrite/ops/cpuOps.h"
#include <cmath>        // expf
#include <functional>   // std::function
#include <limits>       // std::numeric_limits

namespace cpu_ops{
    void matMul(float* C, const float* A, const float* B, int rows, int cols, int inner){
        for (int i = 0; i < rows * cols; i++) C[i] = 0.0f;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < inner; k++) {
                    C[i * cols + j] += A[i * inner + k] * B[k * cols + j];
                }
            }
        }

    }

    void matAdd(float* C, const float* A, const float* B, int size){
        for (int i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
    }
    void matSub(float* C, const float* A, const float* B, int size){
        for (int i = 0; i < size; i++) {
            C[i] = A[i] - B[i];
        }
    }

    // ── Elementwise operations ────────────────────────────────────────────────
    void scalarMul(float* out, const float* in, float scalar, int size){
        for(int i = 0; i < size; i++){
            out[i] = in[i] * scalar;
        }
    }
    void scalarAdd(float* out, const float* in, float scalar, int size){
        for(int i = 0; i < size; i++){
            out[i] = in[i] + scalar;
        }
    }
    void elementMul(float* C, const float* A, const float* B, int size){
        for(int i = 0; i < size; i++){
            C[i] = A[i] * B[i];
        }
    }
    void elementDiv(float* C, const float* A, const float* B, int size){
        for(int i = 0; i < size; i++){
            C[i] = A[i] / B[i];
        }
    }

    // ── Shape operations ──────────────────────────────────────────────────────
    void transpose(float* out, const float* in, int rows, int cols){
        for(int i = 0; i < rows; i ++){
            for(int j = 0; j < cols; j ++){
                out[j * rows+ i] = in[j + cols * i];
            }
        }
    }
    void broadcast(float* out, const float* in, int rows, int out_cols){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < out_cols; j++){
                out[i * out_cols + j] = in[i];
            }
        }
    }

    // ── Reductions ────────────────────────────────────────────────────────────
    void colSum(float* out, const float* in, int rows, int cols){
        for(int i = 0; i < rows; i ++){
            out[i] = 0.0f;
            for(int j = 0; j < cols; j++){
                out[i] += in[i * cols + j];
            }
        }
    }
    float sum(const float* in, int size){
        float result = 0;
        for(int i = 0; i < size; i++){
            result += in[i];
        }
        return result;
    }

    void softmax(float* out, const float* in, int rows, int cols){
        for (int j = 0; j < cols; j++) {
            float denom = 0.0f;
            for (int i = 0; i < rows; i++)
                denom += expf(in[i * cols + j]);
            for (int i = 0; i < rows; i++)
                out[i * cols + j] = expf(in[i * cols + j]) / denom;
        }
    }


    // ── Convolution ───────────────────────────────────────────────────────────
    void im2col(float* out, const float* input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols){
        for(int n = 0; n < N; n++){
            for(int out_row = 0; out_row < outH; out_row++){

                for(int out_col = 0; out_col < outW; out_col++){

                    int result_col = n * outH * outW + out_row * outW + out_col;

                    for(int out_c = 0; out_c < in_C; out_c++){
                        
                        for(int kernel_row = 0; kernel_row < kH; kernel_row++){

                            int ih = out_row * stride - pad + kernel_row;

                            for(int kernel_col = 0; kernel_col < kW; kernel_col++){

                                int iw = out_col * stride - pad + kernel_col;

                                int result_row = out_c * kH * kW + kernel_row * kW + kernel_col;

                                if(ih >= 0 && ih < in_H && iw >= 0 && iw < in_W){
                                    out[result_row * n_cols + result_col] = input[(out_c * in_H * in_W + ih * in_W + iw) * N + n];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void col2im(float* grad_input, const float* cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int n_cols){
        for(int n = 0; n < N; n++){
            for(int out_row = 0; out_row < outH; out_row++){

                for(int out_col = 0; out_col < outW; out_col++){

                    int result_col = n * outH * outW + out_row * outW + out_col;

                    for(int out_c = 0; out_c < in_C; out_c++){

                        for(int kernel_row = 0; kernel_row < kH; kernel_row++){

                            int ih = out_row * stride - pad + kernel_row;

                            for(int kernel_col = 0; kernel_col < kW; kernel_col++){

                                int iw = out_col * stride - pad + kernel_col;

                                int result_row = out_c * kH * kW + kernel_row * kW + kernel_col;

                                if(ih >= 0 && ih < in_H && iw >= 0 && iw < in_W){
                                    grad_input[(out_c * in_H * in_W + ih * in_W + iw) * N + n] += cols[result_row * n_cols + result_col];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Misc ──────────────────────────────────────────────────────────────────
    void funcApply(float* out, const float* in, int size, const std::function<float(float)>& fn){
        for(int i = 0; i < size; i++){
            out[i] = fn(in[i]);
        }
    }

    void scatterAdd(float* target, const int* indices, const float* source, int size){
        for (int i = 0; i < size; i++) {
            target[indices[i]] += source[i];
        }
    }

    void relu(float* out, const float* in, int size){
    for (int i = 0; i < size; i++) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }

    void reluBackward(float* out, const float* grad, const float* in, int size){
        for (int i = 0; i < size; i++) out[i] = in[i] > 0.0f ? grad[i] : 0.0f;
    }

    int argmax(const float* in, int size){
        float max_val = std::numeric_limits<float>::lowest();  // need these
        int max_idx = 0;
        for (int i = 0; i < size; i++) {
            if (in[i] > max_val) {
                max_val = in[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    void colSelect(float* out, const float* in, int rows, int in_cols, const int* indices, int n_cols){
        for(int j = 0; j < n_cols; j++){
            int col = indices[j];
            for(int i = 0; i < rows; i++){
                out[i * n_cols + j] = in[i * in_cols + col];
            }
        }
    }

    void reorder(float* out, const float* in, const int* dst_to_src, int size){
        for (int dst = 0; dst < size; dst++) {
            out[dst] = in[dst_to_src[dst]];
        }    
    }

}
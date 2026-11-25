#include <stdio.h>
#include <cuda_runtime.h>
/*
 * conv_ops.cu — CUDA kernels for im2col and col2im convolution primitives
 *
 * PURPOSE:
 *   These two kernels replace the CPU nested-loop implementations in tensor.cu.
 *   Together they allow Conv2D forward and backward passes to run entirely on
 *   the GPU by converting the convolution problem into a matrix multiply problem.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SHARED PARAMETERS (both kernels):
 *
 *   in_C   — number of input channels (e.g. 1 for grayscale, 3 for RGB)
 *   in_H   — input feature map height (pixels)
 *   in_W   — input feature map width (pixels)
 *   kH     — kernel/filter height
 *   kW     — kernel/filter width
 *   stride — step size the kernel moves across the input
 *   pad    — zero-padding added to each spatial edge of the input
 *   outH   — output feature map height = (in_H + 2*pad - kH) / stride + 1
 *   outW   — output feature map width  = (in_W + 2*pad - kW) / stride + 1
 *   N      — batch size (number of images processed simultaneously)
 *   col_len  — number of rows in the im2col matrix = in_C * kH * kW
 *              (one row = one scalar position inside a receptive field, across all channels)
 *   n_cols   — number of columns in the im2col matrix = outH * outW * N
 *              (one column = one receptive field for one sample at one spatial location)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * im2colKernel:
 *
 *   INPUT  — `input`:    the raw image batch, stored as [in_C*in_H*in_W, N] row-major.
 *                        Innermost dimension is batch: input[(c*in_H*in_W + h*in_W + w)*N + n]
 *                        gives pixel at channel c, row h, column w, sample n.
 *
 *   OUTPUT — `out`:      the im2col matrix, stored as [col_len, n_cols] row-major.
 *                        Each column is one flattened receptive field for one sample.
 *                        Each row is one (channel, kernel_row, kernel_col) position
 *                        across every spatial location and every sample.
 *                        Out-of-bounds (padded) positions are written as 0.0f.
 *
 *   GOAL:  After this kernel, multiplying the weight matrix W [out_C, col_len]
 *          by `out` [col_len, n_cols] produces the full conv output in one matmul.
 *          One thread computes one element of `out`.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * col2imKernel:
 *
 *   INPUT  — `cols`:       gradient with respect to the im2col matrix,
 *                          shape [col_len, n_cols] row-major. This is the upstream
 *                          gradient after the backward matmul (W^T * grad_output).
 *
 *   OUTPUT — `grad_input`: accumulated gradient with respect to the original input,
 *                          shape [in_C*in_H*in_W, N] row-major, initialised to zero.
 *                          Because each input pixel may appear in multiple receptive
 *                          fields, multiple threads write to the same location —
 *                          atomicAdd is required to avoid race conditions.
 *
 *   GOAL:  Scatter gradients from the im2col domain back into the input image domain.
 *          This is the exact inverse index mapping of im2colKernel.
 *          One thread reads one element of `cols` and atomicAdds it into `grad_input`.
 */

__global__ void im2colKernel(float* out, const float* input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int col_len, int n_cols, int out_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < out_size){
        int out_row = i / n_cols;
        int out_col = i % n_cols;

        int inChannel = out_row / (kH * kW);

        int k  = out_row % (kH * kW);   // position within kernel
        int kh = k / kW;
        int kw = k % kW; 

        int inBatch = out_col / (outH * outW);
        int p  = out_col % (outH * outW);
        int ph = p / outW;  // output spatial row
        int pw = p % outW;  // output spatial col
        
        int ih = ph * stride - pad + kh;
        int iw = pw * stride - pad + kw;

        if (ih >= 0 && ih < in_H && iw >= 0 && iw < in_W)
            out[i] = input[(inChannel * in_H * in_W + ih * in_W + iw) * N + inBatch];
        else
            out[i] = 0.0f;
    }

}


__global__ void col2imKernel(float* grad_input, const float* cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int outH, int outW, int N, int col_len, int n_cols, int out_size){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < out_size){
        
        int out_row = i / n_cols;
        int out_col = i % n_cols;

        int inChannel = out_row / (kH * kW);

        int k  = out_row % (kH * kW);   // position within kernel
        int kh = k / kW;
        int kw = k % kW; 

        int inBatch = out_col / (outH * outW);
        int p  = out_col % (outH * outW);
        int ph = p / outW;  // output spatial row
        int pw = p % outW;  // output spatial col
        
        int ih = ph * stride - pad + kh;
        int iw = pw * stride - pad + kw;


        if (ih >= 0 && ih < in_H && iw >= 0 && iw < in_W)
            atomicAdd(&grad_input[(inChannel * in_H * in_W + ih * in_W + iw) * N + inBatch], cols[i]);

    }
}
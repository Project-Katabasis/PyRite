#include "pyrite/tensor.h"
#include "pyrite/layers/conv2d.h"
#include <cmath>
#include <vector>


Conv2D::Conv2D(int in_C, int in_H, int in_W, int out_C, int kH, int kW, int stride, int pad): in_C(in_C), in_H(in_H), in_W(in_W), 
out_C(out_C), kH(kH), kW(kW), stride(stride), pad(pad), W(Tensor::random({out_C, in_C*kH*kW}, -xavierRange(in_C*kH*kW, out_C), xavierRange(in_C*kH*kW, out_C))),
b(Tensor::zeros({out_C, 1})), dW(Tensor::zeros({out_C, in_C*kH*kW})), db(Tensor::zeros({out_C, 1})) {};

float Conv2D::xavierRange(int input_size, int output_size) const {
    return sqrt(6.0f / (input_size + output_size));
}

Tensor Conv2D::forward(Tensor input){
    //std::cout << "Conv2D input shape: " << input.shape[0] << " " << input.shape[1] << std::endl;
    im2col_cache = Tensor::im2col(input, in_C, in_H, in_W, kH, kW, stride, pad);
    Tensor matmul_out = Tensor::matAdd(Tensor::matMul(W, im2col_cache), Tensor::broadcast(b, im2col_cache.shape[1]));
    int outH = (in_H + 2*pad - kH) / stride + 1;
    int outW = (in_W + 2*pad - kW) / stride + 1;
    int N = input.shape[1];
    std::vector<float> reordered(out_C * outH * outW * N);
    for(int c = 0; c < out_C; c++)
        for(int n = 0; n < N; n++)
            for(int ph = 0; ph < outH; ph++)
                for(int pw = 0; pw < outW; pw++){
                    int src = c * (outH*outW*N) + n * outH*outW + ph * outW + pw;
                    int dst = (c * outH*outW + ph*outW + pw) * N + n;
                    reordered[dst] = matmul_out.data[src];
                }
    return Tensor(reordered, {out_C * outH * outW, N});
}

Tensor Conv2D::backward(Tensor grad){
    int outH = (in_H + 2*pad - kH) / stride + 1;
    int outW = (in_W + 2*pad - kW) / stride + 1;
    int N = grad.shape[1];

    // reverse reorder: {out_C*outH*outW, N} → {out_C, outH*outW*N}
    std::vector<float> reordered(out_C * outH * outW * N);
    for(int c = 0; c < out_C; c++)
        for(int n = 0; n < N; n++)
            for(int ph = 0; ph < outH; ph++)
                for(int pw = 0; pw < outW; pw++){
                    int src = (c * outH*outW + ph*outW + pw) * N + n;
                    int dst = c * (outH*outW*N) + n*outH*outW + ph*outW + pw;
                    reordered[dst] = grad.data[src];
                }
    Tensor grad_reordered(reordered, {out_C, outH*outW*N});

    dW = Tensor::matMul(grad_reordered, Tensor::transpose(im2col_cache));
    db = Tensor::colSum(grad_reordered);
    Tensor d_cols = Tensor::matMul(Tensor::transpose(W), grad_reordered);
    return Tensor::col2im(d_cols, in_C, in_H, in_W, kH, kW, stride, pad, im2col_cache.shape[1] / (outH * outW));
}

std::vector<std::pair<Tensor*, Tensor*>> Conv2D::parameters(){
    return {{&W, &dW}, {&b, &db}};
}

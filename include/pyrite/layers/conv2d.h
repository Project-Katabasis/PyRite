#pragma once
#include "pyrite/tensor.h"
#include "pyrite/layers/layer.h"
#include <vector>

class Conv2D : public Layer {
public:
    Conv2D(int in_C, int in_H, int in_W, int out_C, int kH, int kW, int stride=1, int pad=0);
    Tensor forward(Tensor input) override;
    Tensor backward(Tensor grad) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
private:
    int in_C, in_H, in_W, out_C, kH, kW, stride, pad;
    Tensor W, b, dW, db;
    Tensor im2col_cache;
    float xavierRange(int input_size, int output_size) const;
};
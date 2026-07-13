#pragma once
#include "pyrite/tensor.h"
#include "pyrite/layers/layer.h"
#include <vector>


class MaxPool2D : public Layer {
public:
    MaxPool2D(int in_C, int in_H, int in_W, int kH, int kW);
    Tensor forward(Tensor input) override;
    Tensor backward(Tensor grad) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
private:
    int in_C, in_H, in_W, kH, kW;
    std::vector<int> argmax_cache;
    std::vector<int> input_shape_cache;
};
#pragma once
#include "pyrite/tensor.h"
class Layer {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual ~Layer() = default;
    virtual std::vector<std::pair<Tensor*, Tensor*>> parameters() = 0;
};
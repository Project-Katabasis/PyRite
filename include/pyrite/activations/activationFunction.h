#pragma once
#include "pyrite/tensor.h"
#include "pyrite/layers/layer.h"

class ActivationFunction: public Layer {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual ~ActivationFunction() = default;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override {
    return {};
    }
};
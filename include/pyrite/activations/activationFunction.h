#pragma once
#include "pyrite/tensor.h"
class ActivationFunction {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual ~ActivationFunction() = default;
};
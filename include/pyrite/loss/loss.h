#pragma once
#include "pyrite/tensor.h"

class Loss {
public:
    virtual Tensor forward(Tensor prediction, Tensor target) = 0;
    virtual Tensor backward() = 0;
    virtual ~Loss() = default;
};
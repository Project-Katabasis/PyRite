#pragma once
#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/layer.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<Layer*>& layers) = 0;
};
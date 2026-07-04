#pragma once
#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"

class SGD : public Optimizer{
public:
    SGD(float learningRate);
    void step(std::vector<Layer*>& layers) override;

private:
    float lr;
};
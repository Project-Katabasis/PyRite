#pragma once
#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"

class SGDMomentum : public Optimizer{
public:
    SGDMomentum(float learningRate);
    SGDMomentum(float learningRate, float momentumFactor);

    void step(std::vector<Layer*>& layers) override;

private:
    float lr;
    float mf;
    std::vector<Tensor> momentumVect;
};
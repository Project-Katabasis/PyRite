#pragma once
#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"

class Adam : public Optimizer{
public:
    Adam();
    Adam(float momentumFactor, float betaOne, float betaTwo, float epsilon);

    void step(std::vector<Layer*>& layers) override;

private:
    float lr;
    float betaOne;
    float betaTwo;
    float epsilon;
    int t;
    std::vector<Tensor> momentumVect;
    std::vector<Tensor> velocityVect;
};
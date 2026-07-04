#pragma once
#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"

class ReLU : public ActivationFunction {
    public:
        ReLU();
        Tensor forward(Tensor input) override;
        Tensor backward(Tensor grad) override;

    private:
        Tensor input_cache;
};
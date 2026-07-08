#pragma once
#include "pyrite/tensor.h"
#include "pyrite/loss/loss.h"

class SoftmaxCrossEntropy : public Loss {
public:
    SoftmaxCrossEntropy();
    Tensor forward(Tensor prediction, Tensor target) override;
    Tensor backward() override;
    virtual ~SoftmaxCrossEntropy() = default;

private:
    Tensor softmax_cache;
    Tensor target_cache;
};
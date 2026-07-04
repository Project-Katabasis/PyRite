#pragma once
#include "pyrite/tensor.h"
#include "pyrite/loss/loss.h"

class MSE : public Loss{
public:
    MSE();
    virtual Tensor forward(Tensor prediction, Tensor target) ;
    virtual Tensor backward();
    virtual ~MSE() = default;

private:
    Tensor prediction_cache;
    Tensor target_cache;
    int n;
};
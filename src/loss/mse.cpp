#include "pyrite/tensor.h"
#include "pyrite/loss/mse.h"


MSE::MSE() : prediction_cache(Tensor::zeros({1})), target_cache(Tensor::zeros({1})) {}

Tensor MSE::forward(Tensor prediction, Tensor target){
    if(prediction.shape != target.shape){
        throw std::invalid_argument("Prediction shape does not match target shape");
    }
    n = prediction.size();
    prediction_cache = prediction;
    target_cache = target;

    Tensor diff = Tensor::matSub(target, prediction);
    Tensor square = Tensor::elementMul(diff, diff);
    return Tensor::scalarMul(Tensor::sum(square), 1.0f / n);
}

Tensor MSE::backward(){
    return Tensor::scalarMul(Tensor::matSub(prediction_cache, target_cache), 2.0f / n);
}
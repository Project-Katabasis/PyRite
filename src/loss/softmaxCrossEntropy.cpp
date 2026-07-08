#include "pyrite/tensor.h"
#include "pyrite/loss/softmaxCrossEntropy.h"
#include <cmath>


SoftmaxCrossEntropy::SoftmaxCrossEntropy() : softmax_cache(Tensor::zeros({1})), target_cache(Tensor::zeros({1})) {}

Tensor SoftmaxCrossEntropy::forward(Tensor prediction, Tensor target){
    if(prediction.shape != target.shape){
        throw std::invalid_argument("Prediction shape does not match target shape");
    }
    target_cache = target;

    float softmax_denom = Tensor::sum(Tensor::funcApply(prediction, [](float v) -> float { return expf(v); })).data[0];
    Tensor softmax_out = Tensor::funcApply(prediction, [softmax_denom](float v) -> float { return expf(v) / softmax_denom; });

    softmax_cache = softmax_out;

    return Tensor::sum(Tensor::elementMul(target, Tensor::funcApply(softmax_out, [](float v) -> float { return(-std::log(v)) ;})));
}

Tensor SoftmaxCrossEntropy::backward(){
    return Tensor::matSub(softmax_cache, target_cache);
}
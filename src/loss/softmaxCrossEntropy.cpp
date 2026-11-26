#include "pyrite/tensor.h"
#include "pyrite/loss/softmaxCrossEntropy.h"
#include <cmath>


SoftmaxCrossEntropy::SoftmaxCrossEntropy() : softmax_cache(Tensor::zeros({1})), target_cache(Tensor::zeros({1})) {}

Tensor SoftmaxCrossEntropy::forward(Tensor prediction, Tensor target) {
    Tensor softmax_out = Tensor::softmax(prediction);
    softmax_cache = softmax_out;
    target_cache = target;
    Tensor log_s = Tensor::funcApply(softmax_out, [](float v) -> float { return logf(v); });
    return Tensor::scalarMul(Tensor::sum(Tensor::elementMul(target, log_s)), -1.0f / prediction.shape[1]);
}

Tensor SoftmaxCrossEntropy::backward() {
    int cols = softmax_cache.shape[1];
    return Tensor::scalarMul(Tensor::matSub(softmax_cache, target_cache), 1.0f / cols);
}
#include "pyrite/tensor.h"
#include "pyrite/activations/relu.h"


ReLU::ReLU() : input_cache(Tensor::zeros({1})) {}

Tensor ReLU::forward(Tensor input){
    input_cache = input;
    return Tensor::funcApply(input, [](float v) { return v > 0.0f ? v : 0.0f;});
}

Tensor ReLU::backward(Tensor grad){
    Tensor mask = Tensor::funcApply(input_cache, [](float v) { return v > 0 ? 1.0f : 0.0f;});
    return Tensor::elementMul(grad, mask);
}
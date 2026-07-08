#include "pyrite/tensor.h"
#include "pyrite/activations/sigmoid.h"
#include <cmath>

Sigmoid::Sigmoid() : input_cache(Tensor::zeros({1})) {}

Tensor Sigmoid::forward(Tensor input){
    input_cache = input;
    return Tensor::funcApply(input, [](float v) { return (1.0f / (1.0f + expf(-v)));});
}

Tensor Sigmoid::backward(Tensor grad){
    Tensor mask = Tensor::funcApply(input_cache, [](float v) { float s = 1.0f / (1.0f + expf(-v)); return s * (1.0f - s);});
    return Tensor::elementMul(grad, mask);
}
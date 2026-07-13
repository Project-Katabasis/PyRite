#include "pyrite/tensor.h"
#include "pyrite/layers/flatten.h"

Flatten::Flatten() {};

Tensor Flatten::forward(Tensor input){
    input_shape_cache = input.shape;
    int N = input.shape[input.shape.size() - 1];  // last dim is always batch
    int features = input.size() / N;
    return Tensor(input.data, {features, N});
}

Tensor Flatten::backward(Tensor grad){
    return Tensor(grad.data, input_shape_cache);
}

std::vector<std::pair<Tensor*, Tensor*>> Flatten::parameters(){
    return {};
}
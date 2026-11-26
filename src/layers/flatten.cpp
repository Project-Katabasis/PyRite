#include "pyrite/tensor.h"
#include "pyrite/layers/flatten.h"

Flatten::Flatten() {};

Tensor Flatten::forward(Tensor input){
    input_shape_cache = input.shape;
    int N = input.shape[input.shape.size() - 1];  // last dim is always batch
    int features = input.size() / N;
    std::vector<float> data(input.data_ptr(), input.data_ptr() + input.size());
    return Tensor(data, {features, N});
}

Tensor Flatten::backward(Tensor grad){
    std::vector<float> data(grad.data_ptr(), grad.data_ptr() + grad.size());
    return Tensor(data, input_shape_cache);
}

std::vector<std::pair<Tensor*, Tensor*>> Flatten::parameters(){
    return {};
}
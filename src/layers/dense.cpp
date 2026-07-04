#include "pyrite/tensor.h"
#include "pyrite/layers/dense.h"
#include <random>
#include <numeric>
#include <functional>
#include <cmath>
#include <stdexcept>

Dense::Dense(int input_size, int output_size)
    :inputSize(input_size), outputSize(output_size), 
    W(Tensor::random({output_size, input_size}, -xavierRange(input_size, output_size), xavierRange(input_size, output_size))), 
    b(Tensor::zeros({output_size, 1})), input_cache(Tensor::zeros({1})), 
    dW(Tensor::zeros({output_size, input_size})), db(Tensor::zeros({output_size, 1})){}

Dense::Dense(int input_size, int output_size, float wMin, float wMax, float bMin, float bMax)
    :inputSize(input_size), outputSize(output_size), W(Tensor::random({output_size, input_size}, wMin, wMax)), 
    b(Tensor::random({output_size}, bMin, bMax)), input_cache(Tensor::zeros({1})), dW(Tensor::zeros({output_size, input_size})), db(Tensor::zeros({output_size, 1})){}

Dense::Dense(int input_size, int output_size, Tensor weights, Tensor biases)
    :inputSize(input_size), outputSize(output_size), W(weights), b(biases), input_cache(Tensor::zeros({1})),
    dW(Tensor::zeros({output_size, input_size})), db(Tensor::zeros({output_size, 1})){}


float Dense::xavierRange(int input_size, int output_size) const {
    return sqrt(6.0f / (input_size + output_size));
}

Tensor Dense::forward(Tensor input){
    if(input.shape[0] != inputSize){
        throw std::invalid_argument("incorrect Input size in forward");
    }
    input_cache = input;
    return Tensor::matAdd(Tensor::matMul(W, input), Tensor::broadcast(b, input.shape[1]));
}

Tensor Dense::backward(Tensor grad){
    int batch_size = (int)input_cache.shape[1];
    dW = Tensor::scalarMul(Tensor::matMul(grad, Tensor::transpose(input_cache)), 1.0f / batch_size);
    db = Tensor::scalarMul(Tensor::colSum(grad), 1.0f / batch_size);
    return Tensor::matMul(Tensor::transpose(W), grad);
}

std::vector<std::pair<Tensor*, Tensor*>> Dense::parameters() {
    return {{&W, &dW}, {&b, &db}};
}
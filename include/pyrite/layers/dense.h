#pragma once
#include "pyrite/layers/layer.h"
#include "pyrite/tensor.h"


class Dense : public Layer {
public:
    Dense(int input_size, int output_size);
    Dense(int input_size, int output_size, float wMin, float wMax, float bMin, float bMax);
    Dense(int input_size, int output_size, Tensor weights, Tensor biases);

    Tensor forward(Tensor input) override;
    Tensor backward(Tensor grad) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
    
private:
    // friend class Optimizer;  // uncomment when optimizer is implemented
    int inputSize;
    int outputSize;
    int batchSize;
    Tensor W;
    Tensor b;
    Tensor input_cache;
    Tensor dW;
    Tensor db;
    float xavierRange(int input_size, int output_size) const;
};
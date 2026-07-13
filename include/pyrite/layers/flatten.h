#pragma once
#include "pyrite/tensor.h"
#include "pyrite/layers/layer.h"
#include <vector>

class Flatten: public Layer {
public:
    Flatten();
    Tensor forward(Tensor input) override;
    Tensor backward(Tensor grad) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
private:
    std::vector<int> input_shape_cache;
};
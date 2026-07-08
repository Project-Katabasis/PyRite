#include "pyrite/tensor.h"
#include "pyrite/loss/softmaxCrossEntropy.h"
#include <cmath>


SoftmaxCrossEntropy::SoftmaxCrossEntropy() : softmax_cache(Tensor::zeros({1})), target_cache(Tensor::zeros({1})) {}

Tensor SoftmaxCrossEntropy::forward(Tensor prediction, Tensor target) {
    int rows = prediction.shape[0];
    int cols = prediction.shape[1];
    
    // softmax per column
    std::vector<float> s_data(rows * cols);
    for (int j = 0; j < cols; j++) {
        float denom = 0.0f;
        for (int i = 0; i < rows; i++)
            denom += expf(prediction.data[i * cols + j]);
        for (int i = 0; i < rows; i++)
            s_data[i * cols + j] = expf(prediction.data[i * cols + j]) / denom;
    }
    Tensor softmax_out(s_data, {rows, cols});
    
    softmax_cache = softmax_out;
    target_cache = target;
    
    // cross-entropy averaged over batch
    Tensor log_s = Tensor::funcApply(softmax_out, [](float v) -> float { return logf(v); });
    return Tensor::scalarMul(Tensor::sum(Tensor::elementMul(target, log_s)), -1.0f / cols);
}
Tensor SoftmaxCrossEntropy::backward() {
    int cols = softmax_cache.shape[1];
    return Tensor::scalarMul(Tensor::matSub(softmax_cache, target_cache), 1.0f / cols);
}
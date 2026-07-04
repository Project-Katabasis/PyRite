#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/optimizers/sgd.h"

SGD::SGD(float LearningRate): lr(LearningRate){};

void SGD::step(std::vector<Layer*>& layers){
    for(Layer* layer : layers){
        for(auto& [weight, grad] : layer->parameters()){
            *weight = Tensor::matSub(*weight, Tensor::scalarMul(*grad, lr));
        }
    }
}
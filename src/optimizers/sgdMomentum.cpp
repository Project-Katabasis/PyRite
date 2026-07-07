#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/optimizers/sgdMomentum.h"

SGDMomentum::SGDMomentum(float LearningRate): lr(LearningRate), mf(0.9){};

SGDMomentum::SGDMomentum(float LearningRate, float momentumFactor): lr(LearningRate), mf(momentumFactor){};


void SGDMomentum::step(std::vector<Layer*>& layers){
    if(momentumVect.empty()){
    // first call — initialize one zero velocity per parameter
    for(Layer* layer : layers)
        for(auto& [weight, grad] : layer->parameters())
            momentumVect.push_back(Tensor::zeros(weight->shape));
    }
    
    int idx = 0;
    for(Layer* layer : layers){
        for(auto& [weight, grad] : layer->parameters()){
            momentumVect[idx] = Tensor::matAdd(
                Tensor::scalarMul(momentumVect[idx], mf),
                *grad
            );
            *weight = Tensor::matSub(*weight, Tensor::scalarMul(momentumVect[idx], lr));
            idx++;
        }
    }
}
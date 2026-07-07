#include "pyrite/tensor.h"
#include <vector>
#include <cmath>
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/optimizers/adam.h"

Adam::Adam(): lr(0.001f), betaOne(0.9f), betaTwo(0.999f), epsilon(1e-8f), t(0) {}; //default values taken from Keras

Adam::Adam(float learningRate, float betaOne, float betaTwo, float epsilon): lr(learningRate), betaOne(betaOne), betaTwo(betaTwo),
epsilon(epsilon), t(0)  {};


void Adam::step(std::vector<Layer*>& layers){
    if(momentumVect.empty()){
    // first call — initialize one zero velocity per parameter
        for(Layer* layer : layers){
            for(auto& [weight, grad] : layer->parameters()){
                momentumVect.push_back(Tensor::zeros(weight->shape));
                velocityVect.push_back(Tensor::zeros(weight->shape));
            }
        }
    }

    int idx = 0;
    t++;
    for(Layer* layer : layers){
        for(auto& [weight, grad] : layer->parameters()){
            momentumVect[idx] = Tensor::matAdd(Tensor::scalarMul(momentumVect[idx], betaOne), Tensor::scalarMul(*grad, (1-betaOne)));
            velocityVect[idx] = Tensor::matAdd(Tensor::scalarMul(velocityVect[idx], betaTwo), Tensor::scalarMul(Tensor::elementMul(*grad, *grad),(1-betaTwo)));

            Tensor m_hat = Tensor::scalarMul(momentumVect[idx], 1.0f / (1 - pow(betaOne, t)));
            Tensor v_hat = Tensor::scalarMul(velocityVect[idx], 1.0f / (1 - pow(betaTwo, t)));

            *weight = Tensor::matSub(*weight, Tensor::scalarMul(Tensor::elementDiv(m_hat, Tensor::funcApply(v_hat, std::function<float(float)>([this](float v){ return sqrtf(v) + epsilon; }))), lr));
            idx++;
        }
    }
}
#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"
#include "pyrite/layers/layer.h"
#include "pyrite/loss/loss.h"
#include "pyrite/network.h"


Network::Network(): loss(nullptr), epochs(0) {}

void Network::add(Layer* layer){
    layers.push_back(layer);
}

void Network::set_loss(Loss* Loss){
    loss = Loss;
}

void Network::set_epochs(int Epochs){
    epochs = Epochs;
}

Tensor Network::forward(Tensor input){
    Tensor next_input = input;
    for(int i = 0; i < layers.size(); i++){
        next_input = layers[i]->forward(next_input);
    }
    return next_input;
}

void Network::backward(){
    Tensor grad = loss->backward();
    for(int i = layers.size() -1 ; i >= 0; i--){
        grad = layers[i]->backward(grad);
    }
}
#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"
#include "pyrite/layers/layer.h"
#include "pyrite/loss/loss.h"
#include "pyrite/network.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/dataLoader.h"

Network::Network(): loss(nullptr), epochs(0), optimizer(nullptr), dataLoader(nullptr){}

void Network::add(Layer* layer){
    layers.push_back(layer);
}

void Network::set_loss(Loss* Loss){
    loss = Loss;
}

void Network::set_epochs(int Epochs){
    epochs = Epochs;
}

void Network::set_optimizer(Optimizer* Optimizer){
    optimizer = Optimizer;
}

void Network::set_dataLoader(DataLoader* DataLoader){
    dataLoader = DataLoader;
}

Tensor Network::forward(Tensor input){
    Tensor next_input = input;
    for(int i = 0; i < (int)layers.size(); i++){
        next_input = layers[i]->forward(next_input);
    }
    return next_input;
}

void Network::backward(){
    Tensor grad = loss->backward();
    for(int i = (int)layers.size() -1 ; i >= 0; i--){
        grad = layers[i]->backward(grad);
    }
}

void Network::step(){
    optimizer->step(layers);
}

void Network::train(Tensor input, Tensor target){
    Tensor prediction = forward(input);
    loss->forward(prediction, target);
    backward();
    step();
}

void Network::trainEpochs(){
    for(int i = 0; i < epochs; i++){
        while(dataLoader->has_next()){
            auto [input, target] = dataLoader->next();
            train(input, target);
        }
        dataLoader->reset();
    }
}

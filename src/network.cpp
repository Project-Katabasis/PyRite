#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"
#include "pyrite/layers/layer.h"
#include "pyrite/loss/loss.h"
#include "pyrite/network.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/dataLoader.h"
#include <iostream>
#include <chrono>

Network::Network(): loss(nullptr), epochs(0), optimizer(nullptr), dataLoader(nullptr), verbose(false){}

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

void Network::set_verbose(bool v){ verbose = v; }

std::vector<float> Network::getLosses(){ return loss_history; }

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

Tensor Network::train(Tensor input, Tensor target){
    Tensor prediction = forward(input);
    Tensor loss_val = loss->forward(prediction, target);
    backward();
    step();
    return loss_val;
}

void Network::trainEpochs(){
    auto total_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < epochs; i++){
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        int batches = 0;
        while(dataLoader->has_next()){
            auto [input, target] = dataLoader->next();
            epoch_loss += train(input, target).data[0];
            batches++;
        }
        loss_history.push_back(epoch_loss / batches);
        if(verbose){
            auto epoch_end = std::chrono::high_resolution_clock::now();
            float seconds = std::chrono::duration<float>(epoch_end - epoch_start).count();
            std::cout << "Epoch " << i+1 << "/" << epochs 
                      << "  loss: " << epoch_loss / batches 
                      << "  time: " << seconds << "s" << std::endl;
        }
        dataLoader->reset();
    }
    if(verbose){
        auto total_end = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float>(total_end - total_start).count();
        std::cout << "Total training time: " << total << "s" << std::endl;
    }
}
#include <iostream>
#include <stdexcept>
#include <vector>
#include "pyrite/tensor.h"
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/optimizers/sgd.h"
#include "pyrite/layers/dense.h"
#include "pyrite/network.h"
#include "pyrite/activations/relu.h"
#include "pyrite/loss/mse.h"

int main() {
    Network network;

    network.add(new Dense(2,4));
    network.add(new ReLU());
    network.add(new Dense(4, 1));

    network.set_loss(new MSE());
    network.set_optimizer(new SGD(0.01f));


    std::vector<Tensor> inputs = {};
    inputs.push_back(Tensor({1.0f, 0.0f}, {2, 1}));
    inputs.push_back(Tensor({0.0f, 1.0f}, {2, 1}));
    inputs.push_back(Tensor({0.0f, 0.0f}, {2, 1}));
    inputs.push_back(Tensor({1.0f, 1.0f}, {2, 1}));

    std::vector<Tensor> targets = {};
    targets.push_back(Tensor({1.0f}, {1, 1}));
    targets.push_back(Tensor({1.0f}, {1, 1}));
    targets.push_back(Tensor({0.0f}, {1, 1}));
    targets.push_back(Tensor({0.0f}, {1, 1}));

    int Epochs = 10000;

    for(int i = 0; i < Epochs; i++){
        for(size_t j = 0; j < inputs.size(); j++){
            network.train(inputs[j], targets[j]);
        }
    }

    std::cout << "\n--- Final Predictions ---" << std::endl;
    for(size_t j = 0; j < inputs.size(); j++){
        Tensor pred = network.forward(inputs[j]);
        std::cout << "[" << inputs[j].data[0] << ", " << inputs[j].data[1] << "]"
                << " -> " << pred.data[0]
                << " (target: " << targets[j].data[0] << ")" << std::endl;
    }

}



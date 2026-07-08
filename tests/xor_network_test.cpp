#include <iostream>
#include <stdexcept>
#include <vector>
#include "pyrite/tensor.h"
#include "pyrite/layers/layer.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/optimizers/sgd.h"
#include "pyrite/optimizers/sgdMomentum.h"
#include "pyrite/optimizers/adam.h"
#include "pyrite/layers/dense.h"
#include "pyrite/network.h"
#include "pyrite/activations/relu.h"
#include "pyrite/activations/sigmoid.h"
#include "pyrite/loss/mse.h"

int main() {
    Network network;

    network.add(new Dense(2,4));
    network.add(new Sigmoid());
    //network.add(new ReLU());
    network.add(new Dense(4, 1));

    network.set_loss(new MSE());

    //network.set_optimizer(new SGD(0.01f));

    //network.set_optimizer(new SGDMomentum(0.001f));
    network.set_optimizer(new Adam());

    // std::vector<Tensor> inputs = {};
    // inputs.push_back(Tensor({1.0f, 0.0f}, {2, 1}));
    // inputs.push_back(Tensor({0.0f, 1.0f}, {2, 1}));
    // inputs.push_back(Tensor({0.0f, 0.0f}, {2, 1}));
    // inputs.push_back(Tensor({1.0f, 1.0f}, {2, 1}));
    

    // std::vector<Tensor> targets = {};
    // targets.push_back(Tensor({1.0f}, {1, 1}));
    // targets.push_back(Tensor({1.0f}, {1, 1}));
    // targets.push_back(Tensor({0.0f}, {1, 1}));
    // targets.push_back(Tensor({0.0f}, {1, 1}));

    // shape {2, 4} — each column is one sample
    Tensor inputs({1.0f, 0.0f,
                0.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 1.0f}, {2, 4});

    // shape {1, 4}
    Tensor targets({1.0f, 1.0f, 0.0f, 0.0f}, {1, 4});



    int Epochs = 10000;

    // for(int i = 0; i < Epochs; i++){
    //     for(size_t j = 0; j < inputs.size(); j++){
    //         network.train(inputs[j], targets[j]);
    //     }
    // }
    

    for(int i = 0; i < Epochs; i++){
        network.train(inputs, targets);
    }

    // std::cout << "\n--- Final Predictions ---" << std::endl;
    // for(size_t j = 0; j < inputs.size(); j++){
    //     Tensor pred = network.forward(inputs[j]);
    //     std::cout << "[" << inputs[j].data[0] << ", " << inputs[j].data[1] << "]"
    //             << " -> " << pred.data[0]
    //             << " (target: " << targets[j].data[0] << ")" << std::endl;
    // }

    std::cout << "\n--- Final Predictions ---" << std::endl;
    Tensor pred = network.forward(inputs);
    pred.print();
}



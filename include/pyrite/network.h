#pragma once
#include <vector>

#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"
#include "pyrite/layers/layer.h"
#include "pyrite/loss/loss.h"
#include "pyrite/optimizers/optimizer.h"

class Network{
    public:
        Network();

        void add(Layer* layer);
        
        void set_optimizer(Optimizer* Optimizer);

        void set_loss(Loss* Loss);

        void set_epochs(int Epochs);

        Tensor forward(Tensor input);

        void backward();

        void step();

    private:
        std::vector<Layer*> layers;
        Loss* loss;
        int epochs;
        Optimizer* optimizer;
    };
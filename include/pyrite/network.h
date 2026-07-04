#pragma once
#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"
#include "pyrite/layers/layer.h"
#include "pyrite/loss/loss.h"
#include <vector>
class Network{
    public:
        Network();

        void add(Layer* layer);
        
        //set_optimizer(Optimizer optimizer);

        void set_loss(Loss* Loss);

        void set_epochs(int Epochs);

        Tensor forward(Tensor input);

        void backward();

    private:
        std::vector<Layer*> layers;
        Loss* loss;
        int epochs;
    };
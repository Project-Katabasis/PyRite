#pragma once
#include <vector>

#include "pyrite/tensor.h"
#include "pyrite/activations/activationFunction.h"
#include "pyrite/layers/layer.h"
#include "pyrite/loss/loss.h"
#include "pyrite/optimizers/optimizer.h"
#include "pyrite/dataLoader.h"

class Network{
    public:
        Network();

        void add(Layer* layer);
        
        void set_optimizer(Optimizer* Optimizer);

        void set_loss(Loss* Loss);

        void set_epochs(int Epochs);

        void set_dataLoader(DataLoader* DataLoader);
        void set_verbose(bool verbose);

        std::vector<float> getLosses();

        Tensor forward(Tensor input);

        void backward();

        void step();

        Tensor train(Tensor input, Tensor target);

        void trainEpochs();

    private:
        std::vector<Layer*> layers;
        Loss* loss;
        int epochs;
        Optimizer* optimizer;
        DataLoader* dataLoader;
        std::vector<float> loss_history;
        std::vector<float> acc_history;
        bool verbose;
    };
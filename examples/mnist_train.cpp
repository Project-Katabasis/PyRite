#include <iostream>
#include "pyrite/tensor.h"
#include "pyrite/layers/dense.h"
#include "pyrite/activations/relu.h"
#include "pyrite/loss/softmaxCrossEntropy.h"
#include "pyrite/optimizers/adam.h"
#include "pyrite/network.h"
#include "pyrite/dataLoader.h"
#include "pyrite/examples/mnist.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
static const Device DEVICE = Device::CUDA;
#else
static const Device DEVICE = Device::CPU;
#endif

float evaluate(Network& net, Tensor& images, Tensor& labels, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        std::vector<int> idx = {i};
        Tensor input = Tensor::colSelect(images, idx);
        Tensor label = Tensor::colSelect(labels, idx);

#ifdef USE_CUDA
        input = input.to(Device::CUDA);
#endif

        Tensor output = net.forward(input);

#ifdef USE_CUDA
        output = output.to(Device::CPU);
        label  = label.to(Device::CPU);
#endif

        if (output.argmax() == label.argmax()) correct++;
    }
    return 100.0f * correct / n_samples;
}

int main() {
    std::cout << "=== PyRite MNIST Training";
#ifdef USE_CUDA
    std::cout << " [CUDA]";
#endif
    std::cout << " ===" << std::endl;

    std::cout << "Loading MNIST..." << std::endl;
    Tensor train_images = loadMNISTImages("data/train-images-idx3-ubyte", 60000);
    Tensor train_labels = loadMNISTLabels("data/train-labels-idx1-ubyte", 60000);
    Tensor test_images  = loadMNISTImages("data/t10k-images-idx3-ubyte",  10000);
    Tensor test_labels  = loadMNISTLabels("data/t10k-labels-idx1-ubyte",  10000);
    std::cout << "Loaded." << std::endl;

    Network net;
    net.add(new Dense(784, 128));
    net.add(new ReLU());
    net.add(new Dense(128, 64));
    net.add(new ReLU());
    net.add(new Dense(64, 10));

    net.set_loss(new SoftmaxCrossEntropy());
    net.set_optimizer(new Adam());

#ifdef USE_CUDA
    net.to(Device::CUDA);
#endif

    DataLoader loader(train_images, train_labels, 32, true);
    net.set_dataLoader(&loader);
    net.set_epochs(15);
    net.set_verbose(true);

    std::cout << "\nTraining..." << std::endl;
    net.trainEpochs();

    float acc = evaluate(net, test_images, test_labels, 2000);
    std::cout << "Test accuracy: " << acc << "%" << std::endl;
    std::cout << "\nDone." << std::endl;
    return 0;
}
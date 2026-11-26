#include <iostream>
#include "pyrite/tensor.h"
#include "pyrite/layers/dense.h"
#include "pyrite/activations/relu.h"
#include "pyrite/loss/softmaxCrossEntropy.h"
#include "pyrite/optimizers/adam.h"
#include "pyrite/network.h"
#include "pyrite/dataLoader.h"
#include "pyrite/examples/mnist.h"

float evaluate(Network& net, Tensor& images, Tensor& labels, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        // extract single sample — shape {784, 1}
        std::vector<int> idx = {i};
        Tensor input = Tensor::colSelect(images, idx);
        Tensor label = Tensor::colSelect(labels, idx);

        Tensor output = net.forward(input);

        // argmax of output
        int pred = output.argmax();
        int truth = label.argmax();

        if (pred == truth) correct++;
    }
    return 100.0f * correct / n_samples;
}

int main() {
    std::cout << "=== PyRite MNIST Training ===" << std::endl;

    // Load data — use max_samples to load a subset for speed
    std::cout << "Loading MNIST..." << std::endl;
    Tensor train_images = loadMNISTImages("data/train-images-idx3-ubyte", 60000);
    Tensor train_labels = loadMNISTLabels("data/train-labels-idx1-ubyte", 60000);
    Tensor test_images  = loadMNISTImages("data/t10k-images-idx3-ubyte",  10000);
    Tensor test_labels  = loadMNISTLabels("data/t10k-labels-idx1-ubyte",  10000);
    std::cout << "Loaded." << std::endl;

    // Build network: 784 -> 128 -> ReLU -> 64 -> ReLU -> 10
    Network net;
    net.add(new Dense(784, 128));
    net.add(new ReLU());
    net.add(new Dense(128, 64));
    net.add(new ReLU());
    net.add(new Dense(64, 10));

    net.set_loss(new SoftmaxCrossEntropy());
    net.set_optimizer(new Adam());

    // DataLoader — batch size 32, shuffle on
    DataLoader loader(train_images, train_labels, 32, true);
    net.set_dataLoader(&loader);
    net.set_epochs(15);
    net.set_verbose(true);

    // Training loop with accuracy reporting each epoch
    std::cout << "\nTraining..." << std::endl;
    net.trainEpochs();
    float acc = evaluate(net, test_images, test_labels, 2000);
    std::cout << "Test accuracy: " << acc << "%" << std::endl;
    std::cout << "\nDone." << std::endl;
    return 0;
}
#include <iostream>
#include "pyrite/tensor.h"
#include "pyrite/layers/dense.h"
#include "pyrite/layers/conv2d.h"
#include "pyrite/layers/maxpool2d.h"
#include "pyrite/layers/flatten.h"
#include "pyrite/activations/relu.h"
#include "pyrite/loss/softmaxCrossEntropy.h"
#include "pyrite/optimizers/adam.h"
#include "pyrite/network.h"
#include "pyrite/dataLoader.h"
#include "pyrite/examples/mnist.h"
#include <chrono>
// make mnist_cnn
//./mnist_cnn
// 
// Architecture:
// Input {784, N}
// Conv2D(1, 28, 28, 8, 3, 3)  -> {8*26*26, N} = {5408, N}
// ReLU
// MaxPool2D(8, 26, 26, 2, 2)  -> {8*13*13, N} = {1352, N}
// Flatten                      -> {1352, N}
// Dense(1352, 64)              -> {64, N}
// ReLU
// Dense(64, 10)                -> {10, N}
// SoftmaxCrossEntropy

float evaluate(Network& net, Tensor& images, Tensor& labels, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        std::vector<int> idx = {i};
        Tensor input  = Tensor::colSelect(images, idx);
        Tensor label  = Tensor::colSelect(labels, idx);
        Tensor output = net.forward(input);

        int pred = output.argmax();
        int truth = label.argmax();

        if (pred == truth) correct++;
    }
    return 100.0f * correct / n_samples;
}

int main() {
    std::cout << "=== PyRite MNIST CNN ===" << std::endl;

    std::cout << "Loading MNIST..." << std::endl;
    Tensor train_images = loadMNISTImages("data/train-images-idx3-ubyte", 60000);
    Tensor train_labels = loadMNISTLabels("data/train-labels-idx1-ubyte", 60000);
    Tensor test_images  = loadMNISTImages("data/t10k-images-idx3-ubyte",  10000);
    Tensor test_labels  = loadMNISTLabels("data/t10k-labels-idx1-ubyte",  10000);
    std::cout << "Loaded." << std::endl;

    Network net;
    net.add(new Conv2D(1, 28, 28, 16, 3, 3));
    net.add(new ReLU());
    net.add(new MaxPool2D(16, 26, 26, 2, 2));
    net.add(new Flatten());
    net.add(new Dense(16*13*13, 64));
    net.add(new ReLU());
    net.add(new Dense(64, 10));

    net.set_loss(new SoftmaxCrossEntropy());
    net.set_optimizer(new Adam());
    net.set_verbose(true);

    DataLoader loader(train_images, train_labels, 32, true);
    net.set_dataLoader(&loader);
    net.set_epochs(10);

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\nTraining..." << std::endl;
    for (int epoch = 0; epoch < 10; epoch++) {
        while (loader.has_next()) {
            auto [input, target] = loader.next();
            net.train(input, target);
        }
        loader.reset();

        float acc = evaluate(net, test_images, test_labels, 2000);
        std::cout << "Epoch " << epoch + 1 << "/10  test accuracy: " << acc << "%" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    float seconds = std::chrono::duration<float>(end - start).count();
    std::cout << "Total training time: " << seconds << "s" << std::endl;

    std::cout << "\nDone." << std::endl;
    return 0;
}
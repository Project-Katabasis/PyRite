#include <iostream>
#include <stdexcept>
#include "pyrite/tensor.h"
#include "pyrite/layers/dense.h"

int main() {
    std::cout << "Dense Layer Tests" << std::endl;

    // Test 1: default constructor (xavier init)
    std::cout << "\n--- Constructor (default xavier init) ---" << std::endl;
    Dense d1(3, 2);

    // Test 2: custom range constructor
    std::cout << "\n--- Constructor (custom range) ---" << std::endl;
    Dense d2(3, 2, -0.5f, 0.5f, 0.0f, 0.0f);

    // Test 3: forward pass
    std::cout << "\n--- Forward ---" << std::endl;
    Tensor input({1.0f, 2.0f, 3.0f}, {3, 1});
    input.print();
    Tensor output = d1.forward(input);
    output.print();

    // Test 4: forward wrong input size
    std::cout << "\n--- Forward wrong size ---" << std::endl;
    try {
        Tensor bad_input({1.0f, 2.0f}, {2, 1});
        d1.forward(bad_input).print();
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }

    // Test 5: backward
    std::cout << "\n--- Backward ---" << std::endl;
    Tensor grad({1.0f, 1.0f}, {2, 1});
    Tensor d_input = d1.backward(grad);
    d_input.print();

    return 0;
}
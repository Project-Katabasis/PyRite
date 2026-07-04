#include <iostream>
#include <stdexcept>
#include "pyrite/tensor.h"
#include "pyrite/activations/relu.h"

int main() {
    std::cout << "ReLU Activation Tests" << std::endl;

    ReLU relu;

    // Test 1: forward — mix of positive and negative
    std::cout << "\n--- Forward (mix) ---" << std::endl;
    Tensor input({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    input.print();
    Tensor output = relu.forward(input);
    output.print();
    // expected: [0, 0, 0, 1, 2]

    // Test 2: forward — all negative
    std::cout << "\n--- Forward (all negative) ---" << std::endl;
    Tensor all_neg({-3.0f, -2.0f, -1.0f}, {3});
    relu.forward(all_neg).print();
    // expected: [0, 0, 0]

    // Test 3: forward — all positive
    std::cout << "\n--- Forward (all positive) ---" << std::endl;
    Tensor all_pos({1.0f, 2.0f, 3.0f}, {3});
    relu.forward(all_pos).print();
    // expected: [1, 2, 3]

    // Test 4: backward — gradient only flows where input was positive
    std::cout << "\n--- Backward ---" << std::endl;
    relu.forward(input);  // re-cache the mix input
    Tensor grad({1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {5});
    Tensor d_input = relu.backward(grad);
    d_input.print();
    // expected: [0, 0, 0, 1, 1]

    // Test 5: backward — with non-uniform grad
    std::cout << "\n--- Backward (non-uniform grad) ---" << std::endl;
    relu.forward(input);
    Tensor grad2({2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {5});
    relu.backward(grad2).print();
    // expected: [0, 0, 0, 5, 6]

    return 0;
}
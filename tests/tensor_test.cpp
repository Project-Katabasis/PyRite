#include <iostream>
#include <stdexcept>
#include "pyrite/tensor.h"

int main() {
    std::cout << "PyRite Tensor Tests" << std::endl;

    // Test 1: manual data + shape
    std::cout << "\n--- Constructor (manual) ---" << std::endl;
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    a.print();

    // Test 2: 1D inferred shape
    std::cout << "\n--- Constructor (1D inferred) ---" << std::endl;
    Tensor b({5.0f, 6.0f, 7.0f});
    b.print();

    // Test 3: random in range
    std::cout << "\n--- Random ---" << std::endl;
    Tensor c = Tensor::random({2, 3}, -1.0f, 1.0f);
    c.print();

    // Test 4: zeros
    std::cout << "\n--- Zeros ---" << std::endl;
    Tensor z = Tensor::zeros({2, 3});
    z.print();

    // Test 5: matAdd
    std::cout << "\n--- matAdd ---" << std::endl;
    Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor y({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Tensor::matAdd(x, y).print();

    // Test 6: matAdd shape mismatch
    std::cout << "\n--- matAdd shape mismatch ---" << std::endl;
    try {
        Tensor bad({1.0f, 2.0f, 3.0f}, {3});
        Tensor::matAdd(x, bad).print();
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }

    // Test 7: matSub
    std::cout << "\n--- matSub ---" << std::endl;
    Tensor::matSub(x, y).print();

    // Test 8: matMul
    std::cout << "\n--- matMul ---" << std::endl;
    Tensor m1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor m2({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Tensor::matMul(m1, m2).print();

    // Test 9: matMul shape mismatch
    std::cout << "\n--- matMul shape mismatch ---" << std::endl;
    try {
        Tensor bad_mul({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
        Tensor::matMul(x, bad_mul).print();  // {2,2} * {3,2} → cols(2) != rows(3)
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }

    // Test 10: transpose
    std::cout << "\n--- transpose ---" << std::endl;
    Tensor t1({0.7f, 4.0f, 6.0f, 0.4f, 1.6f, 8.0f}, {2, 3});
    Tensor::transpose(t1).print();

    // Test 11: scalarMul
    std::cout << "\n--- scalarMul ---" << std::endl;
    Tensor::scalarMul(a, 3.0f).print();

    // Test 12: scalarAdd
    std::cout << "\n--- scalarAdd ---" << std::endl;
    Tensor::scalarAdd(a, 10.0f).print();

    // Test 13: elementMul
    std::cout << "\n--- elementMul ---" << std::endl;
    Tensor::elementMul(x, y).print();

    // Test 14: elementMul shape mismatch
    std::cout << "\n--- elementMul shape mismatch ---" << std::endl;
    try {
        Tensor d({1.0f, 2.0f, 3.0f}, {3});
        Tensor::elementMul(a, d).print();
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }

    // Test 15: sum
    std::cout << "\n--- sum ---" << std::endl;
    Tensor::sum(a).print();

    // Test 16: funcApply square
    std::cout << "\n--- funcApply (square) ---" << std::endl;
    Tensor::funcApply(a, [](float v) { return v * v; }).print();

    // Test 17: funcApply relu preview
    std::cout << "\n--- funcApply (relu preview) ---" << std::endl;
    Tensor r({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor::funcApply(r, [](float v) { return v > 0 ? v : 0.0f; }).print();

    return 0;
}
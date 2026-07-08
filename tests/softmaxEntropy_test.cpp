#include <iostream>
#include <stdexcept>
#include "pyrite/tensor.h"
#include "pyrite/loss/softmaxCrossEntropy.h"

int main() {
    std::cout << "SoftmaxCrossEntropy Loss Tests" << std::endl;

    SoftmaxCrossEntropy sce;

    // Test 1: confident correct prediction — loss should be small (~0.407)
    std::cout << "\n--- Forward (confident correct prediction) ---" << std::endl;
    Tensor logits1({2.0f, 1.0f, 0.0f}, {3, 1});
    Tensor target1({1.0f, 0.0f, 0.0f}, {3, 1});
    Tensor loss1 = sce.forward(logits1, target1);
    loss1.print();
    // softmax([2,1,0]) ≈ [0.665, 0.245, 0.090]
    // loss = -log(0.665) ≈ 0.407

    // Test 2: backward after Test 1 — grad = softmax - target
    std::cout << "\n--- Backward (grad = softmax - target) ---" << std::endl;
    sce.backward().print();
    // expected: [-0.335, 0.245, 0.090]

    // Test 3: uniform logits — loss should be log(C) = log(3) ≈ 1.099
    std::cout << "\n--- Forward (uniform logits) ---" << std::endl;
    Tensor logits2({1.0f, 1.0f, 1.0f}, {3, 1});
    Tensor target2({0.0f, 1.0f, 0.0f}, {3, 1});
    sce.forward(logits2, target2).print();
    // softmax = [0.333, 0.333, 0.333], loss = -log(0.333) ≈ 1.099

    // Test 4: backward after uniform — grad should be softmax - target
    std::cout << "\n--- Backward (uniform logits) ---" << std::endl;
    sce.backward().print();
    // expected: [0.333, -0.667, 0.333]

    // Test 5: shape mismatch throws
    std::cout << "\n--- Shape mismatch (expect exception) ---" << std::endl;
    try {
        Tensor bad({1.0f, 0.0f}, {2, 1});
        Tensor good({1.0f, 0.0f, 0.0f}, {3, 1});
        sce.forward(bad, good);
        std::cout << "ERROR: no exception thrown" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
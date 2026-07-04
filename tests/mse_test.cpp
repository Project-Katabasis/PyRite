#include <iostream>
#include <stdexcept>
#include "pyrite/tensor.h"
#include "pyrite/loss/mse.h"

int main() {
    std::cout << "MSE Loss Tests" << std::endl;

    MSE mse;

    // Test 1: perfect prediction — loss should be 0
    std::cout << "\n--- Forward (perfect prediction) ---" << std::endl;
    Tensor y_hat1({1.0f, 2.0f, 3.0f}, {3, 1});
    Tensor y1({1.0f, 2.0f, 3.0f}, {3, 1});
    Tensor loss1 = mse.forward(y_hat1, y1);
    loss1.print();
    // expected: [0]

    // Test 2: known loss — (1-2)^2 + (2-4)^2 + (3-6)^2 = 1+4+9=14, /3 = 4.667
    std::cout << "\n--- Forward (known error) ---" << std::endl;
    Tensor y_hat2({1.0f, 2.0f, 3.0f}, {3, 1});
    Tensor y2({2.0f, 4.0f, 6.0f}, {3, 1});
    Tensor loss2 = mse.forward(y_hat2, y2);
    loss2.print();
    // expected: [4.667]

    // Test 3: backward after Test 2 — 2*(y_hat - y)/n
    // 2*(1-2)/3, 2*(2-4)/3, 2*(3-6)/3 = -0.667, -1.333, -2.0
    std::cout << "\n--- Backward (after known error) ---" << std::endl;
    Tensor grad = mse.backward();
    grad.print();
    // expected: [-0.667, -1.333, -2.0]

    // Test 4: backward after perfect prediction — grad should be all zeros
    std::cout << "\n--- Backward (after perfect prediction) ---" << std::endl;
    mse.forward(y_hat1, y1);
    mse.backward().print();
    // expected: [0, 0, 0]

    // Test 5: shape mismatch throws
    std::cout << "\n--- Shape mismatch (expect exception) ---" << std::endl;
    try {
        Tensor bad({1.0f, 2.0f}, {2, 1});
        Tensor good({1.0f, 2.0f, 3.0f}, {3, 1});
        mse.forward(bad, good);
        std::cout << "ERROR: no exception thrown" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
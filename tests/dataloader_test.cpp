#include <iostream>
#include <stdexcept>
#include "pyrite/tensor.h"
#include "pyrite/dataLoader.h"

int main() {
    std::cout << "DataLoader Tests" << std::endl;

    // 6 samples, 2 features each — shape {2, 6}
    Tensor inputs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 6});

    // 6 targets, 1 output each — shape {1, 6}
    Tensor targets({0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}, {1, 6});

    // Test 1: batch_size=2, should give 3 batches
    std::cout << "\n--- Batching (batch_size=2, 6 samples = 3 batches) ---" << std::endl;
    DataLoader loader(inputs, targets, 2, false);  // no shuffle for predictable output
    int batch_count = 0;
    while(loader.has_next()){
        auto [input_batch, target_batch] = loader.next();
        std::cout << "Batch " << batch_count << ":" << std::endl;
        input_batch.print();
        target_batch.print();
        batch_count++;
    }
    std::cout << "Total batches: " << batch_count << " (expected 3)" << std::endl;

    // Test 2: reset and verify starts over
    std::cout << "\n--- Reset (should restart from beginning) ---" << std::endl;
    loader.reset();
    std::cout << "has_next after reset: " << (loader.has_next() ? "true" : "false") << " (expected true)" << std::endl;

    // Test 3: shuffle — batches should be in different order each epoch
    std::cout << "\n--- Shuffle (epoch 1) ---" << std::endl;
    DataLoader shuffled(inputs, targets, 2, true);
    while(shuffled.has_next()){
        auto [ib, tb] = shuffled.next();
        tb.print();
    }
    std::cout << "\n--- Shuffle (epoch 2, different order expected) ---" << std::endl;
    shuffled.reset();
    while(shuffled.has_next()){
        auto [ib, tb] = shuffled.next();
        tb.print();
    }

    // Test 4: has_next returns false when exhausted
    std::cout << "\n--- Exhausted loader ---" << std::endl;
    DataLoader small(inputs, targets, 6, false);  // one big batch
    small.next();
    std::cout << "has_next after full pass: " << (small.has_next() ? "true" : "false") << " (expected false)" << std::endl;

    return 0;
}
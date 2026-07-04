#!/bin/bash

if [ -z "$1" ]; then
    echo "Running all tests..."
    make tensor_test && ./tests/tensor_test
    make dense_test && ./tests/dense_test
    make relu_test && ./tests/relu_test
    make mse_test && ./tests/mse_test
    make xor_test && ./tests/xor_test
else
    echo "Running $1 tests..."
    make $1_test && ./tests/$1_test
fi
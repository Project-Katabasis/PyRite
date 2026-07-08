CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -Iinclude

# Source files
SRC = src/tensor.cpp

# Test targets
tensor_test: $(SRC) tests/tensor_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/tensor_test $(SRC) tests/tensor_test.cpp

dense_test: $(SRC) src/layers/dense.cpp tests/dense_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/dense_test $(SRC) src/layers/dense.cpp tests/dense_test.cpp

relu_test: $(SRC) src/activations/relu.cpp tests/relu_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/relu_test $(SRC) src/activations/relu.cpp tests/relu_test.cpp

mse_test: $(SRC) src/loss/mse.cpp tests/mse_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/mse_test $(SRC) src/loss/mse.cpp tests/mse_test.cpp

xor_test: $(SRC) src/layers/dense.cpp src/activations/relu.cpp src/activations/sigmoid.cpp src/loss/mse.cpp src/optimizers/sgd.cpp src/optimizers/sgdMomentum.cpp src/optimizers/adam.cpp src/network.cpp tests/xor_network_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/xor_test $(SRC) src/layers/dense.cpp src/activations/relu.cpp src/activations/sigmoid.cpp src/loss/mse.cpp src/optimizers/sgd.cpp src/optimizers/sgdMomentum.cpp src/optimizers/adam.cpp src/network.cpp tests/xor_network_test.cpp

softmaxCE_test: $(SRC) src/loss/softmaxCrossEntropy.cpp tests/softmaxEntropy_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/softmaxCE_test $(SRC) src/loss/softmaxCrossEntropy.cpp tests/softmaxEntropy_test.cpp

clean:
	rm -f tests/tensor_test tests/dense_test tests/relu_test tests/mse_test
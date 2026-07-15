CXX = nvcc
CXXFLAGS = -std=c++17 -O3 -Iinclude
CUDA_SRCS = src/cuda/matmul.cu src/cuda/elementwise.cu

SRC = src/tensor.cu

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

dataloader_test: $(SRC) src/dataLoader.cpp tests/dataloader_test.cpp
	$(CXX) $(CXXFLAGS) -o tests/dataloader_test $(SRC) src/dataLoader.cpp tests/dataloader_test.cpp

mnist: $(SRC) src/layers/dense.cpp src/activations/relu.cpp src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp src/network.cpp src/dataLoader.cpp examples/mnist.cpp examples/mnist_train.cpp
	$(CXX) $(CXXFLAGS) -o examples/mnist $(SRC) src/layers/dense.cpp src/activations/relu.cpp src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp src/network.cpp src/dataLoader.cpp examples/mnist.cpp examples/mnist_train.cpp

mnist_cnn: examples/mnist_cnn.cpp $(SRC) src/layers/dense.cpp src/layers/conv2d.cpp src/layers/maxpool2d.cpp src/layers/flatten.cpp src/activations/relu.cpp src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp src/network.cpp src/dataLoader.cpp examples/mnist.cpp
	$(CXX) $(CXXFLAGS) -o mnist_cnn examples/mnist_cnn.cpp $(SRC) src/layers/dense.cpp src/layers/conv2d.cpp src/layers/maxpool2d.cpp src/layers/flatten.cpp src/activations/relu.cpp src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp src/network.cpp src/dataLoader.cpp examples/mnist.cpp

mnist_cnn_naive: examples/mnist_cnn.cpp $(SRC) $(CUDA_SRCS) src/layers/dense.cpp src/layers/conv2d.cpp src/layers/maxpool2d.cpp src/layers/flatten.cpp src/activations/relu.cpp src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp src/network.cpp src/dataLoader.cpp examples/mnist.cpp
	$(CXX) $(CXXFLAGS) -DCUDA_NAIVE -DUSE_CUDA -o mnist_cnn_naive examples/mnist_cnn.cpp $(SRC) $(CUDA_SRCS) src/layers/dense.cpp src/layers/conv2d.cpp src/layers/maxpool2d.cpp src/layers/flatten.cpp src/activations/relu.cpp src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp src/network.cpp src/dataLoader.cpp examples/mnist.cpp

clean:
	rm -f tests/tensor_test tests/dense_test tests/relu_test tests/mse_test mnist_cnn mnist_cnn_naive
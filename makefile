CXX      = nvcc
CXXFLAGS = -std=c++17 -O3 -Iinclude

SRC = src/tensor.cu

CUDA_SRC = src/cuda/elementwise.cu src/cuda/matmul.cu src/cuda/conv_ops.cu \
           src/cuda/reductions.cu src/ops/cudaOps.cu \
           src/memory/allocators/cudaAllocator.cu

COMMON_SRC = src/layers/dense.cpp src/layers/conv2d.cpp src/layers/maxpool2d.cpp \
             src/layers/flatten.cpp src/activations/relu.cpp \
             src/loss/softmaxCrossEntropy.cpp src/optimizers/adam.cpp \
             src/network.cpp src/dataLoader.cpp examples/mnist.cpp \
             src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp \
             src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

# ── Tests ─────────────────────────────────────────────────────────────────────
tensor_test: $(SRC) tests/tensor_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)
	$(CXX) $(CXXFLAGS) -o tests/tensor_test $(SRC) tests/tensor_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

dense_test: $(SRC) src/layers/dense.cpp tests/dense_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)
	$(CXX) $(CXXFLAGS) -o tests/dense_test $(SRC) src/layers/dense.cpp tests/dense_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

relu_test: $(SRC) src/activations/relu.cpp tests/relu_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)
	$(CXX) $(CXXFLAGS) -o tests/relu_test $(SRC) src/activations/relu.cpp tests/relu_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

softmaxCE_test: $(SRC) src/loss/softmaxCrossEntropy.cpp tests/softmaxEntropy_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)
	$(CXX) $(CXXFLAGS) -o tests/softmaxCE_test $(SRC) src/loss/softmaxCrossEntropy.cpp tests/softmaxEntropy_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

dataloader_test: $(SRC) src/dataLoader.cpp tests/dataloader_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)
	$(CXX) $(CXXFLAGS) -o tests/dataloader_test $(SRC) src/dataLoader.cpp tests/dataloader_test.cpp src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

kernel_test: tests/kernel_test.cu $(SRC) src/cuda/elementwise.cu src/cuda/matmul.cu src/cuda/conv_ops.cu src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)
	$(CXX) $(CXXFLAGS) -o tests/kernel_test tests/kernel_test.cu $(SRC) src/cuda/elementwise.cu src/cuda/matmul.cu src/cuda/conv_ops.cu src/memory/storage.cpp src/memory/allocators/cpuAllocator.cpp src/ops/ops.cpp src/ops/cpuOps.cpp $(CUDA_SRC)

# ── Examples ──────────────────────────────────────────────────────────────────
mnist_train: examples/mnist_train.cpp $(SRC) $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -o mnist_train examples/mnist_train.cpp $(SRC) $(COMMON_SRC)

mnist_train_cuda: examples/mnist_train.cpp $(SRC) $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -o mnist_train_cuda examples/mnist_train.cpp $(SRC) $(COMMON_SRC)

mnist_cnn: examples/mnist_cnn.cpp $(SRC) $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -o mnist_cnn examples/mnist_cnn.cpp $(SRC) $(COMMON_SRC)

mnist_cnn_cuda: examples/mnist_cnn.cpp $(SRC) $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -o mnist_cnn_cuda examples/mnist_cnn.cpp $(SRC) $(COMMON_SRC)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f tests/tensor_test tests/dense_test tests/relu_test tests/softmaxCE_test \
	      tests/dataloader_test tests/kernel_test \
	      mnist_train mnist_train_cuda mnist_cnn mnist_cnn_cuda
# PyRite

A PyTorch-style deep learning library written from scratch in C++, with Python bindings via pybind11.

PyRite is built as a ground-up implementation of a deep learning framework — tensors, layers, optimizers, GPU acceleration, and a Python API — accompanied by an academic paper covering the mathematics, design decisions, and performance benchmarks.

## Status

**Phase 1 — Core FC Network ✅ Complete**
- [x] Tensor (flat 1D storage, full operation set)
- [x] Dense layer (forward + backward)
- [x] ReLU activation
- [x] MSE loss
- [x] Network (forward, backward, train loop)
- [x] SGD optimizer
- [x] XOR trained to convergence — end-to-end pipeline verified

**Phase 2 — Real Training**
- [x] Batching support
- [x] Adam optimizer
- [x] MNIST training

**Phase 3 — Convolutional (up next)**
- [ ] Conv2D, MaxPool, BatchNorm

**Phase 4 — GPU Acceleration**
- [ ] OpenMP CPU parallelism
- [ ] CUDA kernels
- [ ] Benchmarks vs PyTorch

**Phase 5 — Python Bindings**
- [ ] pybind11 bindings
- [ ] PyPI release

## Build & Test

Requires g++ with C++17.

```bash
# Run all tests
./run_tests.sh

# Run a specific test
./run_tests.sh tensor
./run_tests.sh dense
./run_tests.sh relu
./run_tests.sh mse
```

## Project Structure

```
PyRite/
├── include/pyrite/       # Header files
│   ├── tensor.h
│   ├── layers/
│   ├── activations/
│   └── loss/
├── src/                  # Implementations
├── tests/                # Test files
├── Makefile
└── run_tests.sh
```
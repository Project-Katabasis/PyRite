# PyRite Architecture

A PyTorch-style deep learning library written in C++ with Python bindings via pybind11.
**Goal:** Full implementation of a deep learning framework from scratch — tensors, layers, optimizers, GPU acceleration, and Python bindings. Accompanied by an academic paper covering the math, design, and benchmarks.

---

## Project Vision

One project, one paper. The full arc:
- Phase 1: Core FC network (Tensor, Dense, ReLU, MSE, Network, SGD) — train XOR/simple regression
- Phase 2: Real training (Adam, batching, MNIST)
- Phase 3: Convolutional (Conv2D, MaxPool, BatchNorm, image classification)
- Phase 4: GPU acceleration (CUDA kernels, benchmark vs PyTorch)
- Phase 5: Python bindings (pybind11, pip install pyrite)

Paper targets a student workshop at a venue like NeurIPS once GPU benchmarks are complete.

---

## Design Principles

- Flat 1D storage for all tensors — shape is metadata that describes how to index into it
- Element `[i][j]` in a `{rows, cols}` tensor = `data[i * cols + j]`
- Abstract base classes for Layer and ActivationFunction — polymorphism via `virtual` functions
- Activations are separate layers — not baked into Dense, plug into Network sequence independently
- matMul is strictly 2D — Conv2D handles im2col reshape before calling matMul
- Adrian writes the code himself; Claude helps with architecture, debugging, and parallelization

---

## File Structure

```
PyRite/
├── include/pyrite/
│   ├── tensor.h
│   ├── layers/
│   │   ├── layer.h              ← abstract base
│   │   ├── dense.h
│   │   ├── conv2d.h             ← Phase 3
│   │   ├── maxpool.h            ← Phase 3
│   │   ├── batchnorm.h          ← Phase 3
│   │   └── network.h
│   ├── activations/
│   │   ├── activationFunction.h ← abstract base
│   │   └── relu.h
│   ├── loss/
│   │   ├── loss.h               ← abstract base
│   │   └── mse.h
│   └── optimizers/
│       ├── optimizer.h          ← abstract base
│       ├── sgd.h
│       └── adam.h
├── src/
│   ├── tensor.cpp
│   ├── layers/
│   │   ├── dense.cpp
│   │   ├── conv2d.cpp           ← Phase 3
│   │   ├── maxpool.cpp          ← Phase 3
│   │   ├── batchnorm.cpp        ← Phase 3
│   │   └── network.cpp
│   ├── activations/
│   │   └── relu.cpp
│   ├── loss/
│   │   └── mse.cpp
│   └── optimizers/
│       ├── sgd.cpp
│       └── adam.cpp
└── tests/
    ├── tensor_test.cpp          ← DONE
    ├── dense_test.cpp           ← DONE
    ├── relu_test.cpp            ← DONE
    ├── mse_test.cpp
    ├── network_test.cpp
    ├── sgd_test.cpp
    └── adam_test.cpp
```

---

## Tensor Class

```cpp
class Tensor {
public:
    std::vector<float> data;   // flat 1D storage
    std::vector<int> shape;
    bool requires_grad;

    // Constructors
    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);
    Tensor(std::vector<float> data);   // infers 1D shape

    // Static factory methods
    static Tensor random(std::vector<int> shape, float min = 0.0f, float max = 1.0f);
    static Tensor zeros(std::vector<int> shape);

    // Operations
    static Tensor matAdd(Tensor a, Tensor b);
    static Tensor matSub(Tensor a, Tensor b);
    static Tensor matMul(Tensor a, Tensor b);      // strictly 2D
    static Tensor scalarMul(Tensor tensor, float num);
    static Tensor scalarAdd(Tensor tensor, float num);
    static Tensor elementMul(Tensor a, Tensor b);
    static Tensor transpose(Tensor tensor);
    static Tensor sum(Tensor tensor);
    static Tensor funcApply(Tensor tensor, std::function<float(float)> func);
    static Tensor reshape(Tensor tensor, std::vector<int> newShape);  // Phase 3
    static Tensor im2col(Tensor tensor, int kH, int kW, int stride);  // Phase 3

    void print() const;
    int size() const;
};
```

**Flat index formula:**
- 2D `[i][j]` → `data[i * cols + j]`
- 3D `[i][j][k]` → `data[i * rows * cols + j * cols + k]`

---

## Layer Base Class

```cpp
class Layer {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual ~Layer() = default;
};
```

---

## ActivationFunction Base Class

```cpp
class ActivationFunction {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual ~ActivationFunction() = default;
};
```

---

## Loss Base Class

```cpp
class Loss {
public:
    virtual Tensor forward(Tensor prediction, Tensor target) = 0;
    virtual Tensor backward() = 0;
    virtual ~Loss() = default;
};
```

---

## Dense Layer

- `W` shape: `{output_size, input_size}` — Xavier initialized
- `forward(x)` → `matMul(W, x) + b`
- `backward(grad)` → `dW = grad * x^T`, `db = grad`, returns `W^T * grad`
- Caches input for backward

---

## ReLU

- Stateless except for `input_cache`
- `forward(x)` → `funcApply(x, max(0, x))`
- `backward(grad)` → `elementMul(grad, mask)` where mask is 1 where input > 0

---

## MSE Loss

- `forward(y_hat, y)` → `sum((y_hat - y)^2) / n`
- `backward()` → `2 * (y_hat - y) / n`
- Caches prediction and target for backward

---

## Network

```cpp
class Network {
public:
    std::vector<Layer*> layers;
    void add(Layer* layer);
    Tensor forward(Tensor input);
    void backward(Tensor loss_grad);
    void step(Optimizer& opt);
};
```

- `forward` loops through layers sequentially
- `backward` loops in reverse
- Activations plug in as layers — Network doesn't distinguish

---

## Optimizers

**SGD:** `W = W - lr * dW`

**Adam:** adaptive learning rates with momentum — Phase 2

---

## Conv2D (Phase 3)

- Uses im2col to reshape input into 2D before calling matMul
- Tensor::im2col handles the patch extraction
- Tensor::matMul unchanged — Conv2D feeds it the right shapes
- Tensor::reshape converts output back to 4D

---

## GPU Acceleration (Phase 4)

- CUDA kernels for matMul and conv operations
- OpenMP for CPU parallelism first
- Benchmark against PyTorch on equivalent tasks
- Results go in the paper

---

## Python Bindings (Phase 5)

Via pybind11:
```python
import pyrite
t = pyrite.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
```

---

## Current Status

**Phase 1 — Core FC Network:**
- [x] Tensor: all operations
- [x] Layer base class
- [x] ActivationFunction base class
- [x] Dense layer (forward + backward)
- [x] ReLU (forward + backward)
- [ ] Loss base class
- [ ] MSE loss
- [ ] Network
- [ ] SGD optimizer

**Phase 2 — Real Training:**
- [ ] Adam optimizer
- [ ] Batching
- [ ] Train MNIST

**Phase 3 — Convolutional:**
- [ ] Tensor: reshape, im2col
- [ ] Conv2D
- [ ] MaxPool
- [ ] BatchNorm

**Phase 4 — GPU:**
- [ ] OpenMP parallelization
- [ ] CUDA matMul kernel
- [ ] CUDA conv kernel
- [ ] Benchmarks

**Phase 5 — Python Bindings:**
- [ ] pybind11 setup
- [ ] Expose Tensor, Network, optimizers
- [ ] PyPI publish
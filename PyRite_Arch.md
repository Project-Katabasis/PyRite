# PyRite — Project Bible

A PyTorch-style deep learning library written from scratch in C++ with Python bindings via pybind11.
GitHub: https://github.com/Rig09/PyRite

**Paper title:** "Deep Learning From First Principles: Design, Mathematical Analysis, and C++ Implementation of PyRite"
**Paper target:** Student workshop at a venue like NeurIPS, once GPU benchmarks are complete.

---

## Collaboration Rules

- **Adrian writes all implementation code** (.h and .cpp files in the project)
- **Claude owns:** test files, PYRITE_ARCH.md, paper LaTeX files
- Claude helps with: architecture guidance, debugging, concept explanations, parallelization
- Adrian's goal: understand every single line he writes — this is a deep learning project

---

## Project Vision

One project, one paper. The full arc:
- Phase 1: Core FC network (Tensor, Dense, ReLU, MSE, Network, SGD) — train XOR ✅
- Phase 2: Batching, Adam, MNIST (in progress)
- Phase 3: Convolutional (Conv2D, MaxPool, BatchNorm)
- Phase 4: GPU acceleration (CUDA kernels, benchmark vs PyTorch)
- Phase 5: Python bindings (pybind11, pip install pyrite)

---

## Current Status

**Phase 1 — Core FC Network ✅ COMPLETE**
- [x] Tensor: all operations implemented
- [x] Layer abstract base class
- [x] ActivationFunction abstract base class (extends Layer)
- [x] Loss abstract base class
- [x] Optimizer abstract base class
- [x] Dense layer (forward + backward)
- [x] ReLU (forward + backward)
- [x] MSE loss (forward + backward)
- [x] Network (forward, backward, train, step, train_step)
- [x] SGD optimizer
- [x] XOR trained to convergence — single sample and batched both verified

**Phase 2 — Real Training (IN PROGRESS)**
- [x] Batching — tensors are `{features, batch_size}`, Dense backward averages gradients, bias broadcast added
- [ ] Adam optimizer — next up
- [ ] MNIST data loader + training

**Phase 3 — Convolutional**
- [ ] Tensor: reshape, im2col
- [ ] Conv2D (im2col approach)
- [ ] MaxPool
- [ ] BatchNorm

**Phase 4 — GPU**
- [ ] OpenMP CPU parallelism
- [ ] CUDA matMul kernel
- [ ] CUDA conv kernel
- [ ] Benchmarks vs PyTorch

**Phase 5 — Python Bindings**
- [ ] pybind11 setup
- [ ] Expose Tensor, Network, optimizers
- [ ] PyPI publish

---

## Design Principles

- Flat 1D `std::vector<float>` storage for all tensors — shape is metadata
- Element `[i][j]` in a `{rows, cols}` tensor = `data[i * cols + j]`
- `ActivationFunction` extends `Layer` — activations plug into Network as layers, no special casing
- Optimizer accesses weights via `Layer::parameters()` returning `vector<pair<Tensor*, Tensor*>>` — no friend classes
- matMul is strictly 2D — Conv2D handles im2col reshape before calling it
- Batching: second tensor dimension is batch size. Existing matMul handles it automatically.
- `new Dense(...)` pattern for heap allocation — Network holds `Layer*` pointers

---

## File Structure

```
PyRite/
├── include/pyrite/
│   ├── tensor.h
│   ├── network.h
│   ├── layers/
│   │   ├── layer.h              ← abstract base: forward, backward, parameters
│   │   └── dense.h
│   ├── activations/
│   │   ├── activationFunction.h ← extends Layer, implements parameters() returning {}
│   │   └── relu.h
│   ├── loss/
│   │   ├── loss.h               ← abstract base
│   │   └── mse.h
│   └── optimizers/
│       ├── optimizer.h          ← abstract base: step(vector<Layer*>&)
│       ├── sgd.h
│       └── adam.h               ← Phase 2
├── src/
│   ├── tensor.cpp
│   ├── network.cpp
│   ├── layers/dense.cpp
│   ├── activations/relu.cpp
│   ├── loss/mse.cpp
│   └── optimizers/
│       ├── sgd.cpp
│       └── adam.cpp             ← Phase 2
└── tests/
    ├── tensor_test.cpp          ✅
    ├── dense_test.cpp           ✅
    ├── relu_test.cpp            ✅
    ├── mse_test.cpp             ✅
    └── xor_network_test.cpp     ✅ (batched XOR converges to [1,1,0,0])
```

---

## Tensor Class

```cpp
class Tensor {
public:
    std::vector<float> data;   // flat 1D storage
    std::vector<int> shape;
    bool requires_grad;

    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);
    Tensor(std::vector<float> data);   // infers 1D shape from data.size()

    static Tensor random(std::vector<int> shape, float min = 0.0f, float max = 1.0f);
    static Tensor zeros(std::vector<int> shape);
    static Tensor matAdd(Tensor a, Tensor b);
    static Tensor matSub(Tensor a, Tensor b);
    static Tensor matMul(Tensor a, Tensor b);         // strictly 2D
    static Tensor scalarMul(Tensor tensor, float num);
    static Tensor scalarAdd(Tensor tensor, float num);
    static Tensor elementMul(Tensor a, Tensor b);
    static Tensor transpose(Tensor tensor);
    static Tensor sum(Tensor tensor);                 // returns {1} tensor, double accumulator
    static Tensor funcApply(Tensor tensor, std::function<float(float)> func);
    static Tensor broadcast(Tensor tensor, int cols); // repeats column across batch dim
    static Tensor colSum(Tensor tensor);              // sums each row → {rows, 1}
    static Tensor reshape(Tensor tensor, std::vector<int> newShape);  // Phase 3
    static Tensor im2col(Tensor tensor, int kH, int kW, int stride);  // Phase 3

    void print() const;
    int size() const;
};
```

All operations throw `std::invalid_argument` on shape mismatch (no cerr + dummy returns).

---

## Layer Base Class

```cpp
class Layer {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual std::vector<std::pair<Tensor*, Tensor*>> parameters() = 0;
    virtual ~Layer() = default;
};
```

---

## ActivationFunction Base Class

```cpp
class ActivationFunction : public Layer {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grad) = 0;
    virtual ~ActivationFunction() = default;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override { return {}; }
};
```

Extends Layer — activations have no learnable parameters, `parameters()` returns empty inline.

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

`backward()` takes no args — loss caches prediction and target during forward.

---

## Optimizer Base Class

```cpp
class Optimizer {
public:
    virtual void step(std::vector<Layer*>& layers) = 0;
    virtual ~Optimizer() = default;
};
```

---

## Dense Layer

- `W` shape: `{output_size, input_size}` — Xavier initialized: `sqrt(6 / (in + out))`
- `b` shape: `{output_size, 1}`
- `forward(x)`: validates `input.shape[0] == inputSize`, caches input, returns `matAdd(matMul(W, x), broadcast(b, batch_size))`
- `backward(grad)`:
  - `batch_size = input_cache.shape[1]`
  - `dW = scalarMul(matMul(grad, transpose(input_cache)), 1.0f / batch_size)`
  - `db = scalarMul(colSum(grad), 1.0f / batch_size)`
  - returns `matMul(transpose(W), grad)`
- `parameters()` returns `{{&W, &dW}, {&b, &db}}`

---

## ReLU

- Extends ActivationFunction
- Caches input for backward
- `forward(x)` → `funcApply(x, v > 0 ? v : 0)`
- `backward(grad)` → `elementMul(grad, mask)` where mask is 1 where cached input > 0

---

## MSE Loss

- `forward(y_hat, y)` → `scalarMul(sum(elementMul(diff, diff)), 1.0f / n)` where `diff = matSub(y_hat, y)`
- `backward()` → `scalarMul(matSub(prediction_cache, target_cache), 2.0f / n)`
- Caches: `prediction_cache`, `target_cache`, `int n = prediction.size()`
- `n` naturally scales with batch size — no changes needed for batching

---

## Network

```cpp
class Network {
public:
    Network();
    void add(Layer* layer);
    void set_loss(Loss* loss);
    void set_optimizer(Optimizer* optimizer);
    void set_epochs(int epochs);
    Tensor forward(Tensor input);
    void backward();
    void step();
    Tensor train(Tensor input, Tensor target);   // forward → loss.forward → backward → step, returns loss
private:
    std::vector<Layer*> layers;
    Loss* loss;
    Optimizer* optimizer;
    int epochs;
};
```

- `forward`: loops layers sequentially with `layers[i]->forward`
- `backward`: starts with `loss->backward()`, loops layers in **reverse** with `int i` (NOT size_t — unsigned reverse loop wraps to huge number)
- `step`: calls `optimizer->step(layers)`
- `train`: orchestrates one full training step, returns loss tensor

---

## SGD Optimizer

```cpp
class SGD : public Optimizer {
public:
    SGD(float learningRate);
    void step(std::vector<Layer*>& layers) override;
private:
    float lr;
};
```

`step` loops layers, calls `parameters()` on each, applies `*weight = matSub(*weight, scalarMul(*grad, lr))`.

---

## Batching Design

- Tensor shape convention: `{features, batch_size}` for inputs, `{outputs, batch_size}` for predictions
- matMul handles batching automatically — no changes to matMul
- Dense forward: `broadcast(b, input.shape[1])` expands bias across batch
- Dense backward: averages `dW` and `db` by dividing by `batch_size = input_cache.shape[1]`
- MSE: `n = prediction.size()` naturally grows with batch — already correct
- Training loop: pass whole batch as one tensor, one `network.train(batch_input, batch_target)` call per epoch
- True SGD = batch_size 1, mini-batch GD = batch_size n — same code, different input

---

## Upcoming: Adam Optimizer (Phase 2)

Adam needs per-parameter moment vectors `m` and `v`, and a timestep `t`. Key math:
- `m = β1*m + (1-β1)*grad`
- `v = β2*v + (1-β2)*grad²`
- `m_hat = m / (1 - β1^t)`, `v_hat = v / (1 - β2^t)`
- `w = w - lr * m_hat / (sqrt(v_hat) + ε)`

Defaults: `β1=0.9`, `β2=0.999`, `ε=1e-8`, `lr=0.001`

Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

---

## Upcoming: MNIST (Phase 2)

- Load binary MNIST files in C++ (fixed format: magic number, count, rows, cols, then raw bytes)
- Normalize pixel values: divide by 255.0f
- Input shape: `{784, batch_size}` (28×28 flattened)
- Output shape: `{10, batch_size}` (one-hot or softmax)
- Will need Softmax activation and cross-entropy loss

---

## Conv2D Design (Phase 3)

- im2col approach: reshape input patches into columns, then matMul with weight matrix
- `Tensor::im2col` handles patch extraction → produces 2D matrix
- `Tensor::matMul` unchanged — Conv2D feeds it the right shapes
- `Tensor::reshape` converts output back

---

## GPU Acceleration (Phase 4)

- OpenMP for CPU parallelism first (`-fopenmp` flag)
- CUDA kernels for matMul and conv operations
- Benchmark against PyTorch on equivalent tasks
- Results go in the paper

---

## Python Bindings (Phase 5)

Via pybind11:
```python
import pyrite
net = pyrite.Network()
net.add(pyrite.Dense(2, 4))
```

---

## Paper Outline

Sections planned:
1. Introduction ✅ (written)
2. Background ✅ (partially written — backprop, chain rule derivation started)
3. Implementation — Tensor, Dense, ReLU, MSE, Network (to write)
4. Experiments — XOR, MNIST, GPU benchmarks (to write)
5. Results (to write)
6. Discussion (to write)
7. Conclusion (to write)

---

## Known Gotchas

- `int i` in reverse loops — **never use `size_t`** for reverse iteration (unsigned wraps to huge number, infinite loop + segfault)
- Dense W shape is `{output_size, input_size}` — rows=outputs, cols=inputs (opposite of intuition)
- `broadcast(b, batch_size)` required in Dense forward — bias is `{output, 1}`, output is `{output, batch_size}`
- `Network network()` is a function declaration in C++ (most vexing parse) — use `Network network;`
- `new Dense(2, 4)` not `Dense::Dense(2,4)*` — `new` allocates, calls constructor, returns pointer
- Makefile recipe lines must use tabs not spaces
- WSL git push requires PAT or re-authenticating via VS Code command palette
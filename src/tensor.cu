#include "pyrite/tensor.h"
#include "pyrite/cuda/kernels.cuh"
#include <random>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>
#include "pyrite/memory/allocators/cpuAllocator.h"
#include <cstring>
#include "pyrite/ops/ops.h"
#include "pyrite/ops/cpuOps.h"          // funcApply only — not in ops:: dispatch

static CPUAllocator _cpu_alloc;
Allocator* Tensor::default_alloc_ = &_cpu_alloc;

Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {

        size_t bytes = data.size() * sizeof(float);
        storage_ = std::make_shared<Storage>(bytes, Device::CPU, default_alloc_);
        std::memcpy(storage_->data(), data.data(), bytes);

    }

Tensor::Tensor(std::vector<float> data)
    : shape({(int)data.size()}), requires_grad(false)
{
    size_t bytes = data.size() * sizeof(float);
    storage_ = std::make_shared<Storage>(bytes, Device::CPU, default_alloc_);
    std::memcpy(storage_->data(), data.data(), bytes);
}

Tensor Tensor::random(std::vector<int> shape, float min, float max) {
    int total = 1;
    for (int d : shape) total *= d;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> data(total);
    for (auto& val : data) val = dist(rng);

    return Tensor(data, shape);
}

Tensor Tensor::zeros(std::vector<int> shape) {
    int total = 1;
    for (int d : shape) total *= d;
    std::vector<float> data(total, 0.0f);  // vector of `total` zeros
    return Tensor(data, shape);
}

void Tensor::print() const {
    float* ptr = static_cast<float*>(storage_->data());
    int total = size();
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < total; i++) {
        std::cout << ptr[i];
        if (i < total - 1) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
}

int Tensor::size() const {
    int s = 1;
    for (int d : shape) s *= d;
    return s;
}

Tensor Tensor::matAdd(Tensor a, Tensor b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Shape mismatch in matAdd: incompatible dimensions");
    }
    Tensor result = Tensor::zeros(a.shape); // construct empty result Tensor
    ops::matAdd(result.data_ptr(), a.data_ptr(), b.data_ptr(), a.size(), a.storage_->device());

    return result;
}

Tensor Tensor::matSub(Tensor a, Tensor b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Shape mismatch in matSub: incompatible dimensions");
    }

    Tensor result = Tensor::zeros(a.shape);   // construct empty result Tensor

    ops::matSub(result.data_ptr(), a.data_ptr(), b.data_ptr(), a.size(), a.storage_->device());
    
    return result;
}


Tensor Tensor::matMul(Tensor a, Tensor b) {
    
    if (a.shape.size() < 2 || b.shape.size() < 2 || a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Shape mismatch in matMul: incompatible dimensions");
    }

    // result shape is [a.rows, b.cols]
    int rows = a.shape[0];
    int inner = a.shape[1];  // must equal b.shape[0]
    int cols = b.shape[1];

    Tensor result = Tensor::zeros({rows, cols});

    ops::matMul(result.data_ptr(), a.data_ptr(), b.data_ptr(), rows, cols, inner, a.storage_->device());

    return result;
}

Tensor Tensor::transpose(Tensor tensor){

    // std::vector<float> result(tensor.shape[0]*tensor.shape[1], 0.0f);

    Tensor result = Tensor::zeros({tensor.shape[1], tensor.shape[0]});

    ops::transpose(result.data_ptr(), tensor.data_ptr(), tensor.shape[0], tensor.shape[1], tensor.storage_->device());

    return result;
}

Tensor Tensor::scalarMul(Tensor tensor, float num){

    Tensor result = Tensor::zeros(tensor.shape);

    ops::scalarMul(result.data_ptr(), tensor.data_ptr(), num, tensor.size(), tensor.storage_->device());

    return result;
}


Tensor Tensor::scalarAdd(Tensor tensor, float num){

    Tensor result = Tensor::zeros(tensor.shape);

    ops::scalarAdd(result.data_ptr(), tensor.data_ptr(), num, tensor.size(), tensor.storage_->device());

    return result;
}

Tensor Tensor::elementMul(Tensor a, Tensor b){
    if(a.shape != b.shape){
        throw std::invalid_argument("Shape mismatch in elementMul: Element multiplication must occur between two matrices of equivalent shape");
    }
    
    Tensor result = Tensor::zeros(a.shape);

    ops::elementMul(result.data_ptr(), a.data_ptr(), b.data_ptr(), a.size(), a.storage_->device());

    return result;
}

Tensor Tensor::elementDiv(Tensor a, Tensor b){
    if(a.shape != b.shape){
        throw std::invalid_argument("Shape mismatch in elementDiv: Element division must occur between two matrices of equivalent shape");
    }
    Tensor result = Tensor::zeros(a.shape);

    ops::elementDiv(result.data_ptr(), a.data_ptr(), b.data_ptr(), a.size(), a.storage_->device());

    return result;
}

Tensor Tensor::sum(Tensor tensor){
    float s = ops::sum(tensor.data_ptr(), tensor.size(), tensor.storage_->device());
    return Tensor({s}, {1});
}

Tensor Tensor::funcApply(Tensor tensor, std::function<float(float)> functionApplied){
    Tensor result = Tensor::zeros(tensor.shape);
    cpu_ops::funcApply(result.data_ptr(), tensor.data_ptr(), tensor.size(), functionApplied);
    return result;
}


Tensor Tensor::colSum(Tensor tensor){
    Tensor result = Tensor::zeros({tensor.shape[0], 1});

    ops::colSum(result.data_ptr(), tensor.data_ptr(), tensor.shape[0], tensor.shape[1], tensor.storage_->device());

    return result;
}

Tensor Tensor::broadcast(Tensor tensor, int cols) {
    Tensor result = Tensor::zeros({tensor.shape[0], cols});

    ops::broadcast(result.data_ptr(), tensor.data_ptr(), tensor.shape[0], cols, tensor.storage_->device());
    
    return result;
}

Tensor Tensor::colSelect(Tensor tensor, std::vector<int> indices){
    int rows = tensor.shape[0];
    int n_cols = indices.size();

    Tensor result = Tensor::zeros({rows, n_cols});

    ops::colSelect(result.data_ptr(), tensor.data_ptr(), rows, tensor.shape[1], indices.data(), n_cols, tensor.storage_->device());
    return result;
}

Tensor Tensor::im2col(Tensor input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad){
    int N = input.shape[1];  // not tensor
    int outH = (in_H + 2*pad - kH) / stride + 1;
    int outW = (in_W + 2*pad - kW) / stride + 1;
    int col_len = in_C * kH * kW;
    int n_cols = outH * outW * N;
    
    Tensor result = Tensor::zeros({col_len, n_cols});
    
    ops::im2col(result.data_ptr(), input.data_ptr(), in_C, in_H, in_W, kH, kW, stride, pad, outH, outW, N, n_cols, input.storage_->device());

    return result;
}



Tensor Tensor::col2im(Tensor cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int N){
    int outH = (in_H + 2*pad - kH) / stride + 1;
    int outW = (in_W + 2*pad - kW) / stride + 1;
    //int col_len = in_C * kH * kW;
    int n_cols = outH * outW * N;

    Tensor result = Tensor::zeros({in_C*in_H*in_W, N});
    
    ops::col2im(result.data_ptr(), cols.data_ptr(), in_C, in_H, in_W, kH, kW, stride, pad, outH, outW, N, n_cols, cols.storage_->device());
    
    return result;
}


float Tensor::at(int i) const {
    return data_ptr()[i];
}


// scatter Add
void Tensor::scatterAdd(Tensor& target, const std::vector<int>& indices, const Tensor& source) {
    ops::scatterAdd(target.data_ptr(), indices.data(), source.data_ptr(), indices.size(), target.storage_->device());
}

int Tensor::argmax() const {
    return ops::argmax(data_ptr(), size(), storage_->device());
}

Tensor Tensor::reorder(const Tensor& input, const std::vector<int>& dst_to_src, std::vector<int> shape) {
    Tensor result = Tensor::zeros(shape);
    ops::reorder(result.data_ptr(), input.data_ptr(), dst_to_src.data(), dst_to_src.size(), input.storage_->device());
    return result;
}

Tensor Tensor::softmax(const Tensor& input) {
    Tensor result = Tensor::zeros(input.shape);
    ops::softmax(result.data_ptr(), input.data_ptr(), input.shape[0], input.shape[1], input.storage_->device());
    return result;
}
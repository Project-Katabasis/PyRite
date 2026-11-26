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
    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* a_ptr = static_cast<float*>(a.storage_->data());
    float* b_ptr = static_cast<float*>(b.storage_->data());
    for (int i = 0; i < a.size(); i++) {
        result_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return result;
}

Tensor Tensor::matSub(Tensor a, Tensor b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Shape mismatch in matSub: incompatible dimensions");
    }

    //std::vector<float> result(a.data.size());
    Tensor result = Tensor::zeros(a.shape);   // construct empty result Tensor

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* a_ptr = static_cast<float*>(a.storage_->data());
    float* b_ptr = static_cast<float*>(b.storage_->data());
    for (int i = 0; i < (int)a.size(); i++) {
        // result[i] = a.data[i] - b.data[i];
        result_ptr[i] = a_ptr[i] - b_ptr[i];
    }
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

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* a_ptr = static_cast<float*>(a.storage_->data());
    float* b_ptr = static_cast<float*>(b.storage_->data());

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < inner; k++) {
                result_ptr[i * cols + j] += a_ptr[i * inner + k] * b_ptr[k * cols + j];
            }
        }
    }
    return result;
}

Tensor Tensor::transpose(Tensor tensor){

    // std::vector<float> result(tensor.shape[0]*tensor.shape[1], 0.0f);

    Tensor result = Tensor::zeros({tensor.shape[1], tensor.shape[0]});

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());

    for(int i = 0; i < tensor.shape[0]; i ++){
        for(int j = 0; j < tensor.shape[1]; j ++){
            result_ptr[j * tensor.shape[0]+ i] = tensor_ptr[j + tensor.shape[1] * i];
        }
    }

    return result;
}

Tensor Tensor::scalarMul(Tensor tensor, float num){

    Tensor result = Tensor::zeros(tensor.shape);
    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());

    for (int i = 0; i < tensor.size(); i++) {
        result_ptr[i] = num * tensor_ptr[i];
    }
    return result;
}


Tensor Tensor::scalarAdd(Tensor tensor, float num){

    Tensor result = Tensor::zeros(tensor.shape);
    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());

    for(int i = 0; i < tensor.size(); i++){
        result_ptr[i] = num + tensor_ptr[i];
    }

    return result;
}

Tensor Tensor::elementMul(Tensor a, Tensor b){
    if(a.shape != b.shape){
        throw std::invalid_argument("Shape mismatch in elementMul: Element multiplication must occur between two matrices of equivalent shape");
    }
    
    Tensor result = Tensor::zeros(a.shape);

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* a_ptr = static_cast<float*>(a.storage_->data());
    float* b_ptr = static_cast<float*>(b.storage_->data());


    for(int i = 0; i < a.size(); i++){
        result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
}

Tensor Tensor::elementDiv(Tensor a, Tensor b){
    if(a.shape != b.shape){
        throw std::invalid_argument("Shape mismatch in elementDiv: Element division must occur between two matrices of equivalent shape");
    }
    Tensor result = Tensor::zeros(a.shape);

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* a_ptr = static_cast<float*>(a.storage_->data());
    float* b_ptr = static_cast<float*>(b.storage_->data());

    for(int i = 0; i < a.size(); i++){
        result_ptr[i] = a_ptr[i] / b_ptr[i];
    }
    return result;
}

Tensor Tensor::sum(Tensor tensor){
    double result = 0;

    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());

    for(int i = 0; i < tensor.size(); i++){
        result += tensor_ptr[i];
    }
    return Tensor({(float)result}, {1});
}

Tensor Tensor::funcApply(Tensor tensor, std::function<float(float)> functionApplied){
    Tensor result = Tensor::zeros(tensor.shape);
    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());
    
    for(int i = 0; i < tensor.size(); i++){
        result_ptr[i] = functionApplied(tensor_ptr[i]);
    }
    return result;
}


Tensor Tensor::colSum(Tensor tensor){
    int cols = tensor.shape[1];
    int rows = tensor.shape[0];

    Tensor result = Tensor::zeros({rows, 1});

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());
    
    for(int i = 0; i < rows; i ++){
        for(int j = 0; j < cols; j++){
            result_ptr[i] += tensor_ptr[i * cols + j];
        }
    }
    return result;
}

Tensor Tensor::broadcast(Tensor tensor, int cols) {
    int rows = tensor.shape[0];

    Tensor result = Tensor::zeros({rows, cols});

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result_ptr[i * cols + j] = tensor_ptr[i];
        }
    }
    return result;
}

Tensor Tensor::colSelect(Tensor tensor, std::vector<int> indices){
    int rows = tensor.shape[0];
    int n_cols = indices.size();

    Tensor result = Tensor::zeros({rows, n_cols});

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* tensor_ptr = static_cast<float*>(tensor.storage_->data());

    for(int j = 0; j < n_cols; j++){
        int col = indices[j];
        for(int i = 0; i < rows; i++){
            result_ptr[i * n_cols + j] = tensor_ptr[i * tensor.shape[1] + col];
        }
    }
    return result;
}

Tensor Tensor::im2col(Tensor input, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad){
    int N = input.shape[1];  // not tensor
    int outH = (in_H + 2*pad - kH) / stride + 1;
    int outW = (in_W + 2*pad - kW) / stride + 1;
    int col_len = in_C * kH * kW;
    int n_cols = outH * outW * N;
    
    Tensor result = Tensor::zeros({col_len, n_cols});

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* input_ptr = static_cast<float*>(input.storage_->data());

    for(int n = 0; n < N; n++){
        for(int out_row = 0; out_row < outH; out_row++){

            for(int out_col = 0; out_col < outW; out_col++){

                int result_col = n * outH * outW + out_row * outW + out_col;

                for(int out_c = 0; out_c < in_C; out_c++){
                    
                    for(int kernel_row = 0; kernel_row < kH; kernel_row++){

                        int ih = out_row * stride - pad + kernel_row;

                        for(int kernel_col = 0; kernel_col < kW; kernel_col++){

                            int iw = out_col * stride - pad + kernel_col;

                            int result_row = out_c * kH * kW + kernel_row * kW + kernel_col;

                            if(ih >= 0 && ih < in_H && iw >= 0 && iw < in_W){
                                result_ptr[result_row * n_cols + result_col] = input_ptr[(out_c * in_H * in_W + ih * in_W + iw) * N + n];
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}



Tensor Tensor::col2im(Tensor cols, int in_C, int in_H, int in_W, int kH, int kW, int stride, int pad, int N){
    int outH = (in_H + 2*pad - kH) / stride + 1;
    int outW = (in_W + 2*pad - kW) / stride + 1;
    //int col_len = in_C * kH * kW;
    int n_cols = outH * outW * N;

    Tensor result = Tensor::zeros({in_C*in_H*in_W, N});

    float* result_ptr = static_cast<float*>(result.storage_->data());
    float* cols_ptr = static_cast<float*>(cols.storage_->data());


    for(int n = 0; n < N; n++){
        for(int out_row = 0; out_row < outH; out_row++){

            for(int out_col = 0; out_col < outW; out_col++){

                int result_col = n * outH * outW + out_row * outW + out_col;

                for(int out_c = 0; out_c < in_C; out_c++){

                    for(int kernel_row = 0; kernel_row < kH; kernel_row++){

                        int ih = out_row * stride - pad + kernel_row;

                        for(int kernel_col = 0; kernel_col < kW; kernel_col++){

                            int iw = out_col * stride - pad + kernel_col;

                            int result_row = out_c * kH * kW + kernel_row * kW + kernel_col;

                            if(ih >= 0 && ih < in_H && iw >= 0 && iw < in_W){
                                result_ptr[(out_c * in_H * in_W + ih * in_W + iw) * N + n] += cols_ptr[result_row * n_cols + result_col];
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}
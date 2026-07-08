#include "pyrite/tensor.h"
#include <random>
#include <numeric>
#include <functional>
#include <stdexcept>

Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad)
    : data(data), shape(shape), requires_grad(requires_grad) {}

Tensor::Tensor(std::vector<float> data)
    : data(data), shape({(int)data.size()}), requires_grad(false) {}

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
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i];
        if (i < data.size() - 1) std::cout << ", ";
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

    std::vector<float> result(a.data.size());
    for (int i = 0; i < (int)a.data.size(); i++) {
        result[i] = a.data[i] + b.data[i];
    }
    return Tensor(result, a.shape);
}

Tensor Tensor::matSub(Tensor a, Tensor b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Shape mismatch in matSub: incompatible dimensions");
    }

    std::vector<float> result(a.data.size());
    for (int i = 0; i < (int)a.data.size(); i++) {
        result[i] = a.data[i] - b.data[i];
    }
    return Tensor(result, a.shape);
}


Tensor Tensor::matMul(Tensor a, Tensor b) {
    
    if (a.shape.size() < 2 || b.shape.size() < 2 || a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Shape mismatch in matMul: incompatible dimensions");
    }

    // result shape is [a.rows, b.cols]
    int rows = a.shape[0];
    int inner = a.shape[1];  // must equal b.shape[0]
    int cols = b.shape[1];

    std::vector<float> result(rows * cols, 0.0f);  // flat, initialized to 0

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < inner; k++) {
                result[i * cols + j] += a.data[i * inner + k] * b.data[k * cols + j];
            }
        }
    }
    return Tensor(result, {rows, cols});
}

Tensor Tensor::transpose(Tensor tensor){

    std::vector<float> result(tensor.shape[0]*tensor.shape[1], 0.0f);

    for(int i = 0; i < tensor.shape[0]; i ++){
        for(int j = 0; j < tensor.shape[1]; j ++){
            result[j * tensor.shape[0]+ i] = tensor.data[j + tensor.shape[1] * i];
        }
    }

    return Tensor(result, {tensor.shape[1], tensor.shape[0]});
}

Tensor Tensor::scalarMul(Tensor tensor, float num){

    std::vector<float> result(tensor.size(), 0.0f);

    for(int i = 0; i < tensor.size(); i++){
        result[i] = num * tensor.data[i];
    }

    return Tensor(result, tensor.shape);
}


Tensor Tensor::scalarAdd(Tensor tensor, float num){
    std::vector<float> result(tensor.size(), 0.0f);

    for(int i = 0; i < tensor.size(); i++){
        result[i] = num + tensor.data[i];
    }

    return Tensor(result, tensor.shape);
}

Tensor Tensor::elementMul(Tensor a, Tensor b){
    if(a.shape != b.shape){
        throw std::invalid_argument("Shape mismatch in elementMul: Element multiplication must occur between two matrices of equivalent shape");
    }
    std::vector<float> result(a.size(), 0.0f);

    for(int i = 0; i < a.size(); i++){
        result[i] = a.data[i] * b.data[i];
    }
    return Tensor(result, a.shape);
}

Tensor Tensor::elementDiv(Tensor a, Tensor b){
    if(a.shape != b.shape){
        throw std::invalid_argument("Shape mismatch in elementDiv: Element division must occur between two matrices of equivalent shape");
    }
    std::vector<float> result(a.size(), 0.0f);

    for(int i = 0; i < a.size(); i++){
        result[i] = a.data[i] / b.data[i];
    }
    return Tensor(result, a.shape);
}

Tensor Tensor::sum(Tensor tensor){
    double result = 0;

    for(int i = 0; i < tensor.size(); i++){
        result += tensor.data[i];
    }
    return Tensor({(float)result}, {1});
}

Tensor Tensor::funcApply(Tensor tensor, std::function<float(float)> functionApplied){
    std::vector<float> result(tensor.size(), 0.0f);
    
    for(int i = 0; i < tensor.size(); i++){
        result[i] = functionApplied(tensor.data[i]);
    }
    return Tensor(result, tensor.shape);
}


Tensor Tensor::colSum(Tensor tensor){
    int cols = tensor.shape[1];
    int rows = tensor.shape[0];
    std::vector<float> result(rows, 0.0f);
    for(int i = 0; i < rows; i ++){
        for(int j = 0; j < cols; j++){
            result[i] += tensor.data[i * cols + j];
        }
    }
    return Tensor(result, {rows, 1});
}

Tensor Tensor::broadcast(Tensor tensor, int cols) {
    int rows = tensor.shape[0];
    std::vector<float> result(rows * cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result[i * cols + j] = tensor.data[i];
        }
    }
    return Tensor(result, {rows, cols});
}

Tensor Tensor::colSelect(Tensor tensor, std::vector<int> indices){
    int rows = tensor.shape[0];
    int n_cols = indices.size();
    std::vector<float> result(rows * n_cols);
    for(int j = 0; j < n_cols; j++){
        int col = indices[j];
        for(int i = 0; i < rows; i++){
            result[i * n_cols + j] = tensor.data[i * tensor.shape[1] + col];
        }
    }
    return Tensor(result, {rows, n_cols});
}
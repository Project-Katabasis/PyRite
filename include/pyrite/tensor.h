#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include "pyrite/memory/storage.h"
#include "pyrite/memory/allocator.h"
#include <memory>

class Tensor {
public:
    // std::vector<float> data;
    std::vector<int> shape;
    bool requires_grad;
    float* data_ptr() { return static_cast<float*>(storage_->data()); }
    const float* data_ptr() const { return static_cast<const float*>(storage_->data()); }

    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);
    Tensor(std::vector<float> data);  // infers 1D shape
    Tensor() : shape({0}), requires_grad(false) {storage_ = std::make_shared<Storage>(0, Device::CPU, default_alloc_);}

    static Tensor random(std::vector<int> shape, float min = 0.0f, float max = 1.0f);
    static Tensor zeros(std::vector<int> shape);

    static Tensor matAdd(Tensor a, Tensor b);
    static Tensor matSub(Tensor a, Tensor b);
    static Tensor matMul(Tensor a, Tensor b);
    static Tensor transpose(Tensor tensor);
    static Tensor scalarMul(Tensor tensor, float num);
    static Tensor scalarAdd(Tensor tensor, float num);
    static Tensor elementMul(Tensor a, Tensor b);
    static Tensor elementDiv(Tensor a, Tensor b);
    static Tensor sum(Tensor tensor);
    static Tensor funcApply(Tensor tensor, std::function<float(float)> functionApplied);
    static Tensor colSum(Tensor tensor);
    static Tensor broadcast(Tensor tensor, int cols);
    static Tensor colSelect(Tensor tensor, std::vector<int> indices);
    
    //convolutional functions
    static Tensor im2col(Tensor input, int in_C, int in_H, int in_W,
                        int kH, int kW, int stride, int pad);

    static Tensor col2im(Tensor cols, int in_C, int in_H, int in_W,
                        int kH, int kW, int stride, int pad, int N);

    // element access
    float at(int i) const;
    int argmax() const;

    // scatter
    static void scatterAdd(Tensor& target, const std::vector<int>& indices, const Tensor& source);

    // reorder (conv2d index shuffle)
    static Tensor reorder(const Tensor& input, const std::vector<int>& dst_to_src, std::vector<int> shape);

    // softmax (column-wise)
    static Tensor softmax(const Tensor& input);
        

    void print() const;
    int size() const;

    private:
        std::shared_ptr<Storage> storage_;
        static Allocator* default_alloc_;
};
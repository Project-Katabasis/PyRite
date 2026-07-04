#pragma once
#include <vector>
#include <iostream>
#include <functional>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    bool requires_grad;

    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);
    Tensor(std::vector<float> data);  // infers 1D shape

    static Tensor random(std::vector<int> shape, float min = 0.0f, float max = 1.0f);
    static Tensor zeros(std::vector<int> shape);
    static Tensor matAdd(Tensor a, Tensor b);
    static Tensor matSub(Tensor a, Tensor b);
    static Tensor matMul(Tensor a, Tensor b);
    static Tensor transpose(Tensor tensor);
    static Tensor scalarMul(Tensor tensor, float num);
    static Tensor scalarAdd(Tensor tensor, float num);
    static Tensor elementMul(Tensor a, Tensor b);
    static Tensor sum(Tensor tensor);
    static Tensor funcApply(Tensor tensor, std::function<float(float)> functionApplied);


    void print() const;
    int size() const;
};
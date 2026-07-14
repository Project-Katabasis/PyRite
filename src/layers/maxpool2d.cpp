#include "pyrite/tensor.h"
#include <vector>
#include "pyrite/layers/maxpool2d.h"
#include <limits>

MaxPool2D::MaxPool2D(int in_C, int in_H, int in_W, int kH, int kW): in_C(in_C), in_H(in_H), in_W(in_W), kH(kH), kW(kW) {};

Tensor MaxPool2D::forward(Tensor input){
    //std::cout << "MaxPool input shape: " << input.shape[0] << " " << input.shape[1] << std::endl;
    int N = input.shape[1]; 
    int poolH = in_H / kH;
    int poolW = in_W / kW;
    std::vector<float> result(in_C * poolH * poolW * N, 0.0f);
    argmax_cache.resize(in_C * poolH * poolW * N);

    for(int batch = 0; batch < N; batch ++){
        for(int c = 0; c < in_C; c++){
            for(int poolHCurr = 0; poolHCurr < poolH; poolHCurr++){
                for(int poolWCurr = 0; poolWCurr < poolW; poolWCurr++){
                    float max_val = std::numeric_limits<float>::lowest();
                    int max_idx = -1;
                    for(int curKH = 0; curKH < kH; curKH++){
                        for(int curKW = 0; curKW < kW; curKW++){
                            int input_idx = (c * in_H * in_W + (poolHCurr * kH + curKH) * in_W + (poolWCurr * kW + curKW)) * N + batch;
                            float curr = input.data[input_idx];
                            if(curr > max_val){
                                max_val = curr;
                                max_idx = input_idx;
                            }
                        }
                    }
                    int out_idx = (c * poolH * poolW + poolHCurr * poolW + poolWCurr) * N + batch;
                    result[out_idx] = max_val;
                    argmax_cache[out_idx] = max_idx;
                }
            }
            
        }   
    }
    return Tensor(result, {in_C * poolH * poolW, N});
}

Tensor MaxPool2D::backward(Tensor grad){
    int N = grad.shape[1];
    Tensor grad_input = Tensor::zeros({in_C * in_H * in_W, N});
    for(int i = 0; i < (int)argmax_cache.size(); i++){
        grad_input.data[argmax_cache[i]] += grad.data[i];
    }
    return grad_input;
}

std::vector<std::pair<Tensor*, Tensor*>> MaxPool2D::parameters(){
    return {};
}
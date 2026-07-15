#include <stdio.h>
#include <cuda_runtime.h>


__global__ void elementwiseAdd(float *c, const float *a, const float *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] + b[i];
    }
}
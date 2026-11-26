#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void elementwiseAdd(float *c, const float *a, const float *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] + b[i];
    }
}
__global__ void elementwiseDiv(float *c, const float *a, const float *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = a[i]/b[i];
    }
}

__global__ void elementwiseSub(float *c, const float *a, const float *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] - b[i];
    }
}

__global__ void scalarMulKernel(float *c, const float *a, float scalar, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] * scalar;
    }
}

__global__ void scalarAddKernel(float *c, const float *a, float scalar, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] + scalar;
    }
}

__global__ void elementwiseMulKernel(float *c, float *a, float *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        c[i] = a[i] * b[i];
    }
}

__global__ void elementwiseRelu(float *c, float *a, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size){
        c[i] = fmaxf(a[i], 0.0f);
    }
}

__global__ void reluBackwardKernel(float *c, float *grad, float *a, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < size){
        c[i] = a[i] > 0 ? grad[i] : 0.0f;
    }
}

__global__ void transposeKernel(float *c, float *a, int rows, int cols, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size){
        int row = i / cols;
        int col = i % cols;
        c[col * rows + row] = a[i];
    }
}

__global__ void broadcastKernel(float *c, float *a, int outCols, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size){
        int input = i / outCols;
        c[i] = a[input];
    }
}

__global__ void sqrtKernel(float *c, const float *a, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        c[i] = sqrtf(a[i]);
    }
}
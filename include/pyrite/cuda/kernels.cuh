#pragma once

__global__ void naiveMatMul(float *C, const float *A, const float *B, int m, int n, int k);

__global__ void elementwiseAdd(float *c, const float *a, const float *b, int size);

__global__ void tiledMatMul(float *C, const float *A, const float *B, int m, int n, int k);
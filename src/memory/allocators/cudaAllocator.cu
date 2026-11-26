#include <cstddef>
#include "pyrite/memory/allocator.h"
#include "pyrite/memory/device.h"
#include "pyrite/memory/allocators/cudaAllocator.h"
#include <cstdlib>

void* CUDAAllocator::allocate(size_t bytes){
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}


void CUDAAllocator::deallocate(void* ptr, size_t bytes){
    cudaFree(ptr);
}
#include <cstddef>
#include "pyrite/memory/allocator.h"
#include "pyrite/memory/device.h"
#include "pyrite/memory/allocators/cpuAllocator.h"
#include <cstdlib>
#include <cuda_runtime.h>

void* CPUAllocator::allocate(size_t bytes){
    void* ptr = malloc(bytes);
    return ptr;
}


void CPUAllocator::deallocate(void* ptr, size_t bytes){
    free(ptr);
}
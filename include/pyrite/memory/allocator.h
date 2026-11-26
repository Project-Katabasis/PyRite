#pragma once
#include <cstddef>

class Allocator {
public:
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr, size_t bytes) = 0;
    virtual ~Allocator() = default;
};
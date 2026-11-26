#pragma once
#include <cstddef>
#include "pyrite/memory/allocator.h"
#include "pyrite/memory/device.h"

class CPUAllocator : public Allocator{
    public:
        void* allocate(size_t bytes) override;
        void deallocate(void* ptr, size_t bytes) override;
};
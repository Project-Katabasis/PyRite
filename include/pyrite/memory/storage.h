#pragma once
#include <cstddef>
#include "pyrite/memory/allocator.h"
#include "pyrite/memory/device.h"

class Storage{
    public:
        Storage(size_t nbytes, Device device, Allocator* allocator);

        void* data();
        size_t nbytes();
        Device device();

        ~Storage();

        // no copying — Storage owns its memory
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
    
    private:
        void* ptr_;
        size_t nbytes_;
        Device device_;
        Allocator* allocator_;

};
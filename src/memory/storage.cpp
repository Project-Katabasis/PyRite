#include <cstddef>
#include "pyrite/memory/allocator.h"
#include "pyrite/memory/device.h"
#include "pyrite/memory/storage.h"


Storage::Storage(size_t nbytes, Device device, Allocator* allocator)
    : nbytes_(nbytes), device_(device), allocator_(allocator)
{
    ptr_ = allocator_->allocate(nbytes);
}

void* Storage::data(){
    return ptr_;
}

size_t Storage::nbytes(){
    return nbytes_;
}

Device Storage::device(){
    return device_;
}

Storage::~Storage() {
    allocator_->deallocate(ptr_, nbytes_);
}
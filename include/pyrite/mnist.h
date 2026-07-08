#pragma once
#include <string>
#include "pyrite/tensor.h"

Tensor loadMNISTImages(const std::string& path, int max_samples = -1);
Tensor loadMNISTLabels(const std::string& path, int max_samples = -1);
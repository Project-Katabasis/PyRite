#include "pyrite/mnist.h"
#include <fstream>
#include <stdexcept>
#include <vector>

// MNIST files are big-endian — swap bytes on little-endian machines
static int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) | ((int)c2 << 16) | ((int)c3 << 8) | c4;
}

Tensor loadMNISTImages(const std::string& path, int max_samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open MNIST images file: " + path);

    int magic, n_images, rows, cols;
    file.read((char*)&magic,    4); magic    = reverseInt(magic);
    file.read((char*)&n_images, 4); n_images = reverseInt(n_images);
    file.read((char*)&rows,     4); rows     = reverseInt(rows);
    file.read((char*)&cols,     4); cols     = reverseInt(cols);

    if (magic != 2051)
        throw std::runtime_error("Invalid MNIST images magic number");

    if (max_samples > 0 && max_samples < n_images)
        n_images = max_samples;

    int n_pixels = rows * cols;  // 784 for standard MNIST
    std::vector<float> data(n_pixels * n_images);

    for (int i = 0; i < n_images; i++) {
        for (int j = 0; j < n_pixels; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, 1);
            data[j * n_images + i] = pixel / 255.0f;  // column-major: {784, n_samples}
        }
    }

    return Tensor(data, {n_pixels, n_images});
}

Tensor loadMNISTLabels(const std::string& path, int max_samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open MNIST labels file: " + path);

    int magic, n_labels;
    file.read((char*)&magic,    4); magic    = reverseInt(magic);
    file.read((char*)&n_labels, 4); n_labels = reverseInt(n_labels);

    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST labels magic number");

    if (max_samples > 0 && max_samples < n_labels)
        n_labels = max_samples;

    // one-hot encode: shape {10, n_samples}
    std::vector<float> data(10 * n_labels, 0.0f);

    for (int i = 0; i < n_labels; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        data[(int)label * n_labels + i] = 1.0f;  // column-major one-hot
    }

    return Tensor(data, {10, n_labels});
}
#pragma once
#include "pyrite/tensor.h"
#include <vector>
class DataLoader {
public:
    DataLoader(Tensor inputs, Tensor targets, int batch_size, bool shuffle = true);
    
    // iteration
    void reset();           // call at start of each epoch, reshuffles
    bool has_next();        // are there more batches?
    std::pair<Tensor, Tensor> next();  // returns next batch
    
private:
    Tensor inputs;          // full dataset, shape {features, n_samples}
    Tensor targets;         // full labels, shape {n_classes, n_samples}
    int batch_size;
    bool shuffle;
    int n_samples;
    int current_index;
    std::vector<int> indices;  // shuffled index order
    std::vector<int> get_shuffled_range(int min_val, int max_val); //random order function
};
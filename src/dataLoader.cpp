#include "pyrite/tensor.h"
#include "pyrite/dataLoader.h"
#include <random>
#include <algorithm>
#include <numeric>

DataLoader::DataLoader(Tensor inputs, Tensor targets, int batch_size, bool shuffle ): inputs(inputs), targets(targets), batch_size(batch_size),
 shuffle(shuffle), n_samples(inputs.shape[1]), current_index(0), indices(get_shuffled_range(0, n_samples - 1)) {};

// iteration
void DataLoader::reset(){
    current_index = 0;
    if(shuffle) indices = get_shuffled_range(0, n_samples - 1);
};           // call at start of each epoch, reshuffles


bool DataLoader::has_next(){
    if(current_index + batch_size <= n_samples){
        return true;
    }
    return false;
};

std::vector<int> DataLoader::get_shuffled_range(int min_val, int max_val) {
    if (min_val > max_val) return {};

    std::vector<int> numbers(max_val - min_val + 1);
    
    std::iota(numbers.begin(), numbers.end(), min_val);
    
    thread_local std::mt19937 gen(std::random_device{}());
    
    std::shuffle(numbers.begin(), numbers.end(), gen);
    
    return numbers;
}


std::pair<Tensor, Tensor> DataLoader::next(){
    if(!has_next()){
        throw std::invalid_argument("Already loaded full data set: Reload data before calling again");
    }
    std::vector<int> batch_idx(indices.begin() + current_index, indices.begin() + current_index + batch_size);
    Tensor input_batch = Tensor::colSelect(inputs, batch_idx);
    Tensor target_batch = Tensor::colSelect(targets, batch_idx);
    current_index += batch_size;
    return {input_batch, target_batch};
}; 
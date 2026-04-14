#include "tensor.h"
#include <stdexcept>

float& Tensor::at(uint64_t b, uint64_t i, uint64_t j) {
    if (b >= batch_size || i >= seq_len || j >= dim) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data[b * (seq_len * dim) + i * dim + j];
}

const float& Tensor::at(uint64_t b, uint64_t i, uint64_t j) const {
    if (b >= batch_size || i >= seq_len || j >= dim) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data[b * (seq_len * dim) + i * dim + j];
}

bool is_close(const Tensor& c1, const Tensor& c2, float epsilon) {
    if (c1.size() != c2.size()) return false;

    // epsilon is absolute tolerancy to float accuracy
    return std::equal(c1.begin(), c1.end(), c2.begin(),
        [epsilon](float a, float b) {
            return std::abs(a - b) < epsilon;
        });
}
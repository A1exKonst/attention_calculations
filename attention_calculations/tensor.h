#pragma once
#include <cstdint>
#include <vector>



class Tensor {
private:
    std::vector<float> data;

public:
    uint64_t batch_size;
    uint64_t seq_len;
    uint64_t dim;

    Tensor(const std::vector<float>& data_, 
        uint64_t batch_size_, uint64_t seq_len_, uint64_t dim_) : data(data_), 
        batch_size(batch_size_), seq_len(seq_len_), dim(dim_) {}

    Tensor(std::vector<float>&& data_, 
        uint64_t batch_size_, uint64_t seq_len_, uint64_t dim_) : data(std::move(data_)), 
        batch_size(batch_size_), seq_len(seq_len_), dim(dim_) {}

    Tensor(const Tensor&) = default;

    Tensor(Tensor&&) = default;

    ~Tensor() = default;

    float& at(uint64_t b, uint64_t i, uint64_t j);

    const float& at(uint64_t b, uint64_t i, uint64_t j) const;

    auto begin() { return data.begin(); };

    auto end() { return data.end(); };

    auto begin() const { return data.begin(); };

    auto end() const { return data.end(); };

    size_t size() const { return data.size(); };
};

bool is_close(const Tensor& c1, const Tensor& c2, float epsilon = 1e-5f);
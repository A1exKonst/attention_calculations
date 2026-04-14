#pragma once
#include "tensor.h"

Tensor generate_tensor(unsigned seed = 1, uint64_t batch_size = 10, uint64_t seq_len = 44, uint64_t dim = 34);
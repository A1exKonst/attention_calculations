#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "attention.h"

Tensor attention_with_matmul(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    MatMulType matmul_type
) {
    switch (matmul_type) {
    case MatMulType::NAIVE:
        return naive_attention(Q, K, V);
        break;
    case MatMulType::CACHE_OPTIMIZED:
        return cache_friendly_attention(Q, K, V);
        break;
    case MatMulType::TILING:
        return tiled_attention(Q, K, V, 32);
    case MatMulType::FLASH_ATTENTION:
        return flash_attention(Q, K, V, 32, 32);
    case MatMulType::SIMD:
        return vectorized_attention(Q, K, V, 32);
        break;
    }

    throw std::out_of_range("No such MatMulType found");
}

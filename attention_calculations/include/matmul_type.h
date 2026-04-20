#pragma once

enum class MatMulType {
    NAIVE,
    CACHE_OPTIMIZED,
    TILING,
    SIMD
};
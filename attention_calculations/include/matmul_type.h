#pragma once

enum class MatMulType {
    NAIVE,
    CACHE_OPTIMIZED,
    TILING,
    FLASH_ATTENTION,
    SIMD
};
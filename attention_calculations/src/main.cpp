#include <iostream>
#include <chrono>

#include "tensor.h"
#include "random.h"
#include "attention.h"
#include "matmul_type.h"

int main() {

    Tensor q{ generate_tensor(1, 50, 101, 102) };
    Tensor k{ generate_tensor(2, 50, 101, 102) };
    Tensor v{ generate_tensor(3, 50, 101, 102) };

    auto start = std::chrono::steady_clock::now();
    Tensor naive = attention_with_matmul(q, k, v, MatMulType::NAIVE);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time naive_attention:          " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    Tensor cached = attention_with_matmul(q, k, v, MatMulType::CACHE_OPTIMIZED);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time cache_friendly_attention: " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    Tensor tiled = attention_with_matmul(q, k, v, MatMulType::TILING);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time tiled_attention:          " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    Tensor flash = attention_with_matmul(q, k, v, MatMulType::FLASH_ATTENTION);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time flash_attention:          " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    Tensor vectorized = attention_with_matmul(q, k, v, MatMulType::SIMD);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time vectorized_attention:     " << elapsed.count() << " ms" << std::endl;

    std::cout << std::endl;

    std::cout << "is close(naive, cached):     " << is_close(naive, cached) << std::endl;
    std::cout << "is close(naive, tiled):      " << is_close(naive, tiled) << std::endl;
    std::cout << "is close(naive, flash):      " << is_close(naive, flash) << std::endl;
    std::cout << "is close(naive, vectorized): " << is_close(naive, vectorized) << std::endl;
    

	return 0;
}

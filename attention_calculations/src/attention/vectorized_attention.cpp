#include <immintrin.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "tensor.h"



// documentation:
/*
* __m256 - 256 bit of float values (8 values 32 bit)
* __m128 - 128 bit of float values (4 values 32 bit)
*/



//sum AVX-register and copy result to scalar float register:
inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);                  // interpret __m256 as __m128, does not generate instructions : __m128 _mm256_castps256_ps128 (__m256 a)
    __m128 hi = _mm256_extractf128_ps(v, 1);                // copy upper or lower half into another vector,  generates instructions, such as VEXTRACTF128 : __m128 _mm256_extractf128_ps (__m256, bool upper)
    lo = _mm_add_ps(lo, hi);                                // elementwise addition : __m128 _mm_add_ps(__m128 A, __m128 B)
    __m128 tmp = _mm_add_ps(lo,                             // elementwise addition
        _mm_movehl_ps(lo, lo));                                 // copy high (2 float values) of each argument into new vector : __m128 _mm_movehl_ps(__m128 A, __m128 B)
    tmp = _mm_add_ss(tmp,                                   // sum of a[0] and b[0] as a scalar operation
        _mm_shuffle_ps(tmp, tmp, 1));                           // shuffle 00'00'00'01 -> [F1, F0, F0, F0]
    return _mm_cvtss_f32(tmp);                              // convert into scalar singe precision f32
}

Tensor vectorized_attention(const Tensor& Q, const Tensor& K, const Tensor& V, uint64_t Bc) {
    if (Q.batch_size != K.batch_size ||
        Q.dim != K.dim) {
        throw std::runtime_error("Incompatible Q and K");
    }

    if (K.batch_size != V.batch_size ||
        K.seq_len != V.seq_len) {
        throw std::runtime_error("Incompatible K and V");
    }

    uint64_t batch_size = Q.batch_size;
    uint64_t seq_len_q = Q.seq_len;
    uint64_t seq_len_k = K.seq_len;
    uint64_t d = Q.dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    Tensor output{ std::vector<float>(batch_size * seq_len_q * d, 0.0f), batch_size, seq_len_q, d };

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t i = 0; i < seq_len_q; ++i) {
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;

            for (uint64_t j_start = 0; j_start < seq_len_k; j_start += Bc) {
                uint64_t j_end = std::min(j_start + Bc, seq_len_k);
                uint64_t block_size = j_end - j_start;

                std::vector<float> S_block(block_size);
                float m_block = -std::numeric_limits<float>::infinity();

                // S_block = (Q * K^T) * scale
                for (uint64_t j = j_start; j < j_end; ++j) {
                    __m256 sum_vec = _mm256_setzero_ps();                   // initialize zeroed vector __m256 (8 values of float32)
                    uint64_t k = 0;
                    for (; k + 7 < d; k += 8) {
                        __m256 q_v = _mm256_loadu_ps(&Q.at(b, i, k));       // load unaligned packed single precision (f32)
                        __m256 k_v = _mm256_loadu_ps(&K.at(b, j, k));
                        sum_vec = _mm256_fmadd_ps(q_v, k_v, sum_vec);       // FMA : fused multiply and add: sum_vec = q_v*k_v + sum_vec
                    }
                    float sum = hsum_avx(sum_vec);                          // Previosly defined hsum : horizontal sum of a given vector 

                    // Processing tail from vectorization, vector.size() == d
                    for (; k < d; ++k) {
                        sum += Q.at(b, i, k) * K.at(b, j, k);
                    }
                    // sum = (Q*K^T)[i,j]
                    float score = sum * scale;
                    S_block[j - j_start] = score;
                    m_block = std::max(m_block, score);
                }

                // Online Softmax
                float m_new = std::max(m, m_block);
                float exp_m_diff = std::exp(m - m_new);
                float l_block = 0.0f;
                for (uint64_t j = 0; j < block_size; ++j) {
                    S_block[j] = std::exp(S_block[j] - m_new);
                    l_block += S_block[j];
                }
                float l_new = (l * exp_m_diff) + l_block;

                // Update O_new, O_old:
                float rescale_old = (l * exp_m_diff) / l_new;
                float inv_l_new = 1.0f / l_new;

                __m256 v_rescale_old = _mm256_set1_ps(rescale_old);         // set scalar value to all elements in vector
                __m256 v_inv_l_new = _mm256_set1_ps(inv_l_new);             // set scalar value to all elements in vector

                uint64_t k = 0;
                for (; k + 7 < d; k += 8) {
                    __m256 pv_vec = _mm256_setzero_ps();                    // initialize zeroed vector __m256 (8 values of float32)
                    for (uint64_t j = 0; j < block_size; ++j) {
                        __m256 s_val = _mm256_set1_ps(S_block[j]);          // set scalar value to all elements in vector
                        __m256 v_vec = _mm256_loadu_ps(&V.at(b, j_start + j, k));   // load unaligned packed single precision (f32)
                        pv_vec = _mm256_fmadd_ps(s_val, v_vec, pv_vec);     // FMA : fused multiply and add
                    }

                    __m256 o_vec = _mm256_loadu_ps(&output.at(b, i, k));    // load unaligned packed single precision (f32)
                    // O = O_old * (l*exp/l_new) + (S_block * V_block) / l_new
                    __m256 o_new = _mm256_fmadd_ps(o_vec, v_rescale_old,    // fused multiply and add
                        _mm256_mul_ps(pv_vec, v_inv_l_new));                    // elementwise multiplication
                    _mm256_storeu_ps(&output.at(b, i, k), o_new);           // store unaligned packed single precision
                }

                // Tail || Epilogue (d % 8 != 0):
                for (; k < d; ++k) {
                    float pv_scalar = 0.0f;
                    for (uint64_t j = 0; j < block_size; ++j) {
                        pv_scalar += S_block[j] * V.at(b, j_start + j, k);
                    }
                    float old_val = output.at(b, i, k);
                    output.at(b, i, k) = (old_val * rescale_old) + (pv_scalar * inv_l_new);
                }

                m = m_new;
                l = l_new;
            }
        }
    }
    return output;
}

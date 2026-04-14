#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "attention.h"
#include "tensor.h"
#include "matmul_type.h"

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
        return tiled_attention(Q, K, V, 32);
        break;
    case MatMulType::SIMD:
        return vectorized_attention(Q, K, V, 32);
        break;
    }

    throw std::out_of_range("No such MatMulType found");
}


Tensor naive_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
     uint64_t batch_size = Q.batch_size;
     uint64_t seq_len_q = Q.seq_len;
     uint64_t seq_len_k = K.seq_len;
     uint64_t dk = Q.dim;
     uint64_t dv = V.dim;
     float scale = 1.0f / std::sqrt(static_cast<float>(dk));

     Tensor output{ std::vector<float>(batch_size * seq_len_q * dv, 0.0f), batch_size, seq_len_q, dv };

     for (uint64_t b = 0; b < batch_size; ++b) {
         for (uint64_t i = 0; i < seq_len_q; ++i) {
             std::vector<float> scores(seq_len_k);

             // MatMul(Q, K^T) & Scale
             for (uint64_t j = 0; j < seq_len_k; ++j) {
                 float sum = 0.0f;
                 for (uint64_t l = 0; l < dk; ++l) {
                     sum += Q.at(b, i, l) * K.at(b, j, l);
                 }
                 scores[j] = sum * scale;
             }

             // Softmax
             float max_score = *std::max_element(scores.begin(), scores.end());
             float exp_sum = 0.0f;
             for (float& s : scores) {
                 s = std::exp(s - max_score);
                 exp_sum += s;
             }
             for (float& s : scores) s /= exp_sum;

             // MatMul(Scores, V)
             for (uint64_t j = 0; j < dv; ++j) {
                 float res = 0.0f;
                 for (uint64_t l = 0; l < seq_len_k; ++l) {
                     res += scores[l] * V.at(b, l, j);
                 }
                 output.at(b, i, j) = res;
             }
         }
     }
     return output;
}

Tensor tiled_attention(const Tensor& Q, const Tensor& K, const Tensor& V, uint64_t Bc) {
    uint64_t batch_size = Q.batch_size;
    uint64_t seq_len_q = Q.seq_len;
    uint64_t seq_len_k = K.seq_len;
    uint64_t d = Q.dim; // dk = dv
    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    Tensor O{ std::vector<float>(batch_size * seq_len_q * d, 0.0f), batch_size, seq_len_q, d };

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t i = 0; i < seq_len_q; ++i) {
            float m = -std::numeric_limits<float>::infinity(); // current maximum
            float l = 0.0f;                                    // current sum 

            // Tiling on j:
            for (uint64_t j_start = 0; j_start < seq_len_k; j_start += Bc) {

                uint64_t j_end = std::min(j_start + Bc, seq_len_k);

                // S = Q*K^T
                // m_block is a local max in a block
                std::vector<float> S_block(j_end - j_start);
                float m_block = -std::numeric_limits<float>::infinity();

                for (uint64_t j = j_start; j < j_end; ++j) {
                    float sum = 0.0f;
                    for (uint64_t k = 0; k < d; ++k) {
                        sum += Q.at(b, i, k) * K.at(b, j, k);
                    }
                    S_block[j - j_start] = sum * scale;
                    m_block = std::max(m_block, S_block[j - j_start]);
                }

                // Online Softmax:
                // Update O_new, O_old, m_new, m_old
                float m_new = std::max(m, m_block);
                float exp_m_diff = std::exp(m - m_new);
                float exp_m_block_diff = std::exp(m_block - m_new);

                float l_block = 0.0f;
                for (float& s : S_block) {
                    s = std::exp(s - m_new);
                    l_block += s;
                }

                float l_new = (l * exp_m_diff) + l_block;

                // Update output results according to O_new, O_old
                for (uint64_t k = 0; k < d; ++k) {
                    float pv = 0.0f;
                    for (uint64_t j = j_start; j < j_end; ++j) {
                        pv += S_block[j - j_start] * V.at(b, j, k);
                    }
                    // O_new = O_old * (l*exp / l_new) + (P_block * V_block) / l_new
                    float old_val = O.at(b, i, k);
                    O.at(b, i, k) = (old_val * l * exp_m_diff + pv) / l_new;
                }

                m = m_new;
                l = l_new;
            }
        }
    }
    return O;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "attention.h"
#include "tensor.h"

Tensor tiled_attention(const Tensor& Q, const Tensor& K, const Tensor& V, uint64_t Bc) {
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
    uint64_t dk = Q.dim;
    uint64_t dv = V.dim; // dk = dv
    float scale = 1.0f / std::sqrt(static_cast<float>(dv));

    Tensor output{ batch_size, seq_len_q, dv };

    // output[b, i, j] = sum_l1(sum_l2(Q[b,i,l2]*K[b,l1,l2]*V[b,l1,j]))


    // qk_matmul[l1] = sum_l2(Q[b,i,l2]*K[b,l1,l2])
    // S[l1] = qk_matmul[l1] * scale
    // output[b, i, j] = sum_l1(softmax(S[l1])*V[b,l1,j])

    // iterators:
    // b - parallel. can be tiled, but does not require tiling, because upper dims do not participate in matmul
    // i - parallel.
    // j - parallel.
    // k - reduction.  qk_matmul[i,j] = sum_k(Q[i,k]*K[j,k])

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t i = 0; i < seq_len_q; ++i) {
            float m = -std::numeric_limits<float>::infinity(); // current maximum
            float l = 0.0f;                                    // current sum 

            // Tiling on j:
            for (uint64_t j_start = 0; j_start < seq_len_k; j_start += Bc) {
                uint64_t j_end = std::min(j_start + Bc, seq_len_k);

                // S = Q*K^T * scale
                // m_block is a local max in a block
                std::vector<float> S_block(j_end - j_start);
                float m_block = -std::numeric_limits<float>::infinity();

                for (uint64_t j = j_start; j < j_end; ++j) {

                    // l1 = j - j_start, as S_block index is from zero
                    // qk_matmul[l1] = sum_l2(Q[b,i,l2]*K[b,l1,l2])
                    float sum = 0.0f;
                    for (uint64_t k = 0; k < dv; ++k) {
                        sum += Q(b, i, k) * K(b, j, k);
                    }

                    // S[l1] = qk_matmul[l1] * scale
                    S_block[j - j_start] = sum * scale;
                    m_block = std::max(m_block, sum * scale);
                }

                // Online Softmax:
                // While tiling on j is active, calculate softmax(S[l1])
                // Update O_new, O_old, m_new, m_old
                float m_new = std::max(m, m_block);                 // calculate new maximum "m_new" to avoid overflow in exp(S[l1])
                float exp_m_diff = std::exp(m - m_new);             /* as sum is rescaled to a possible "m_new",
                                                                         we need to calculate exp coefficients for sum of other blocks 
                                                                         and sum of current block */
                float exp_m_block_diff = std::exp(m_block - m_new);

                
                float l_block = 0.0f;                               // l_block is a denominator in a softmax fraction
                for (float& s : S_block) {
                    s = std::exp(s - m_new);
                    l_block += s;
                }

                float l_new = (l * exp_m_diff) + l_block;           /* l_new is a denominator in softmax
                                                                       for the whole tensor row
                                                                       
                                                                       according to m_new, it needs to be rescaled */

                // Update output results according to rescaled O_new, O_old
                // P = softmax(S)
                // pv = (P*V)[i,j]
                float scale_old = l * exp_m_diff / l_new;
                float scale_new = 1.0f / l_new;

                for (uint64_t k = 0; k < dv; ++k) {
                    float pv = 0.0f;
                    for (uint64_t j = j_start; j < j_end; ++j) {
                        pv += S_block[j - j_start] * V(b, j, k);
                    }

                    // O_new = O_old * (l*exp / l_new) + (P_block * V_block) / l_new
                    // O_old (old_val) is an accumulator of softmax result.
                    // 
                    // When summing up new block, O_old should be also rescaled:
                    output(b, i, k) = output(b, i, k) * scale_old + pv * scale_new;
                }
                
                m = m_new;
                l = l_new;
            }
        }
    }
    return output;
}

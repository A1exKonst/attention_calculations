#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "attention.h"

Tensor tiled_attention(const Tensor& Q, const Tensor& K, const Tensor& V, const uint64_t block_size) {
    uint64_t batch_size = Q.batch_size;
    uint64_t seq_len_q = Q.seq_len;
    uint64_t seq_len_k = K.seq_len;
    uint64_t dk = Q.dim;
    uint64_t dv = V.dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dk));

    Tensor output{ batch_size, seq_len_q, dv };

    // output[b, i, j] = sum_l1(sum_l2(Q[b,i,l2]*K[b,l1,l2]*V[b,l1,j]))

    // qk_matmul[l1] = sum_l2(Q[b,i,l2]*K[b,l1,l2])
    // output[b, i, j] = sum_l1(qk_matmul[l1]*V[b,l1,j])

    std::vector<float> qk_matmul(seq_len_k);

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t i = 0; i < seq_len_q; ++i) {

            for (uint64_t l1_block = 0; l1_block < seq_len_k; l1_block += block_size) {
                for (uint64_t l2_block = 0; l2_block < dk; l2_block += block_size) {


                    for (uint64_t l1 = l1_block; l1 < std::min(l1_block + block_size, seq_len_k); ++l1) {
                        float sum = 0.0f;
                        for (uint64_t l2 = l2_block; l2 < std::min(l2_block + block_size, dk); ++l2) {
                            sum += Q(b, i, l2) * K(b, l1, l2);
                            // Q[b,i,l2], K[b,l1,l2] - is already cache-friendly
                        }
                        qk_matmul[l1] = sum * scale;
                    }
                }
            }
            /*
            // MatMul(Q, K^T) * scale
            for (uint64_t l1 = 0; l1 < seq_len_k; ++l1) {
                float sum = 0.0f;
                for (uint64_t l2 = 0; l2 < dk; ++l2) {
                    sum += Q(b, i, l2) * K(b, l1, l2);
                    // Q[b,i,l2], K[b,l1,l2] - is already cache-friendly
                }
                qk_matmul[l1] = sum * scale;
            }
            */
            // Softmax(Q@K^T * scale)
            float max_score = *std::max_element(qk_matmul.begin(), qk_matmul.end());
            float exp_sum = 0.0f;
            for (float& s : qk_matmul) {
                s = std::exp(s - max_score);
                exp_sum += s;
            }
            for (float& s : qk_matmul) s /= exp_sum;

            // MatMul(softmax(Q@K^T * scale), V)
            for (uint64_t l1_block = 0; l1_block < seq_len_k; l1_block += block_size) {
                for (uint64_t j_block = 0; j_block < dv; j_block += block_size) {

                    for (uint64_t l1 = l1_block; l1 < std::min(l1_block + block_size, seq_len_k); ++l1) {
                        for (uint64_t j = j_block; j < std::min(j_block + block_size, dv); ++j) {
                            output(b, i, j) += qk_matmul[l1] * V(b, l1, j);
                        }
                    }
                }
            }
        }
    }
    return output;
}
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "attention.h"

Tensor cache_friendly_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
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
    uint64_t dv = V.dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dk));

    Tensor output{ batch_size, seq_len_q, dv };

    // output[b, i, j] = sum_l1(sum_l2(Q[b,i,l2]*K[b,l1,l2]*V[b,l1,j]))

    // qk_matmul[l1] = sum_l2(Q[b,i,l2]*K[b,l1,l2])
    // output[b, i, j] = sum_l1(qk_matmul[l1]*V[b,l1,j])

    std::vector<float> qk_matmul(seq_len_k);

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t i = 0; i < seq_len_q; ++i) {

            // MatMul(Q, K^T) * scale
            for (uint64_t l1 = 0; l1 < seq_len_k; ++l1) {
                float sum = 0.0f;
                for (uint64_t l2 = 0; l2 < dk; ++l2) {
                    sum += Q(b, i, l2) * K(b, l1, l2);
                    // Q[b,i,l2], K[b,l1,l2] - is already cache-friendly
                }
                qk_matmul[l1] = sum * scale;
            }

            // Softmax(Q@K^T * scale)
            float max_score = *std::max_element(qk_matmul.begin(), qk_matmul.end());
            float exp_sum = 0.0f;
            for (float& s : qk_matmul) {
                s = std::exp(s - max_score);
                exp_sum += s;
            }
            for (float& s : qk_matmul) s /= exp_sum;

            // MatMul(softmax(Q@K^T * scale), V)
            for (uint64_t l1 = 0; l1 < seq_len_k; ++l1) {
                for (uint64_t j = 0; j < dv; ++j) {
                    output(b, i, j) += qk_matmul[l1] * V(b, l1, j);
                    // cycles l1 and j are inverted for cache-friendliness:
                    // qk_matmul[l1]    - is constant relative to j
                    // V[b,l1,j]        - is read in cache rows
                    // output[b,l1,j]   - is written in cache rows
                }
            }
        }
    }
    return output;
}
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "tensor.h"


Tensor naive_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
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

            //std::fill(qk_matmul.begin(), qk_matmul.end(), 0.0);
            //     std::fill not needed, because next cycle rewrites each value in vector

            // MatMul(Q, K^T) & Scale
            for (uint64_t l1 = 0; l1 < seq_len_k; ++l1) {
                float sum = 0.0f;
                for (uint64_t l2 = 0; l2 < dk; ++l2) {
                    sum += Q(b, i, l2) * K(b, l1, l2);
                }
                qk_matmul[l1] = sum * scale;
            }

            // Softmax
            float max_score = *std::max_element(qk_matmul.begin(), qk_matmul.end());
            float exp_sum = 0.0f;
            for (float& s : qk_matmul) {
                s = std::exp(s - max_score);
                exp_sum += s;
            }
            for (float& s : qk_matmul) s /= exp_sum;

            // MatMul(Q@K^T, V)
            for (uint64_t j = 0; j < dv; ++j) {
                float res = 0.0f;
                for (uint64_t l1 = 0; l1 < seq_len_k; ++l1) {
                    res += qk_matmul[l1] * V.at(b, l1, j);
                }
                output(b, i, j) = res;
            }
        }
    }
    return output;
};

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <iostream>

#include "attention.h"

Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V, const size_t block_size_rows, const size_t block_size_columns) {
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
    uint64_t dim_k = Q.dim;
    uint64_t dim_v = V.dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dim_k));

    Tensor output{ batch_size, seq_len_q, dim_v };

    Tensor S{ 1, block_size_rows, block_size_columns };

    Tensor PV{ 1, block_size_rows, dim_v };

    std::vector<float> m(block_size_rows, -std::numeric_limits<float>::infinity());

    std::vector<float> m_block(block_size_rows, -std::numeric_limits<float>::infinity());

    std::vector<float> m_new(block_size_rows, -std::numeric_limits<float>::infinity());

    std::vector<float> l(block_size_rows, 0);

    std::vector<float> l_block(block_size_rows, 0);


    // locking Q[i_block : i_block + Br][0 : d]
    // locking O[i_block : i_block + Br][0 : d] (Tensor output{})
    for (uint64_t b = 0; b < batch_size; ++b) {
        for (size_t i_block = 0; i_block < seq_len_q; i_block += block_size_rows) {
            size_t i_end = std::min(i_block + block_size_rows, seq_len_q);
            size_t local_i_end = std::min(block_size_rows, seq_len_q - i_block);


            // locking K[j_block : j_block + Bc][0 : d]
            // locking V[j_block : j_block + Bc][0 : d]
            for (size_t j_block = 0; j_block < seq_len_k; j_block += block_size_columns) {
                size_t j_end = std::min(j_block + block_size_columns, seq_len_k);
                size_t local_j_end = std::min(block_size_columns, seq_len_k - j_block);


                // 1. calculate S[i_block : i_block + Br][j_block : j_block + Bc]
                // S = Q*K.T
                // S[i, j] = sum_h(Q[i, h] * K[j, h])
                // S[i_block][j_block] clearing not needed, it is fully rewritten
                
                std::fill(m_block.begin(), m_block.end(), -std::numeric_limits<float>::infinity()); // clear m_block from previous block

                for (size_t i = i_block; i < i_end; ++i) {
                    float row_max = -std::numeric_limits<float>::infinity();
                    for (size_t j = j_block; j < j_end; ++j) {
                        float sum = 0;
                        for (size_t h = 0; h < dim_k; ++h) {
                            sum += Q(b, i, h) * K(b, j, h);
                        }
                        S(0, i - i_block, j - j_block) = sum * scale;

                        row_max = std::max(row_max, sum * scale);
                    }

                    m_block[i - i_block] = std::max(m_block[i - i_block], row_max);

                    m_new[i - i_block] = std::max(m_block[i - i_block], m[i - i_block]);
                }
                

                // 2. calculate P[i_block : i_block + Br][j_block : j_block + Bc]
                // after calculating P, matrix S will be no longer needed
                // thats why we will not create P, but will calculate P in place of S:
                // P[i_block][j_block] = exp(S[i_block][j_block] - m_new[i_block])

                // clearing l_block is not needed as it is fully rewritten

                for (size_t i = 0; i < local_i_end; ++i) {
                    float exp_sum = 0;
                    for (size_t j = 0; j < local_j_end; ++j) {
                        S(0, i, j) = std::exp(S(0, i, j) - m_new[i]);
                        exp_sum += S(0, i, j);
                    }

                    l_block[i] = exp_sum;

                    //l_new[i] = l[i] * std::exp(m[i] - m_new[i]) + l_block[i];
                    l[i] = l[i] * std::exp(m[i] - m_new[i]) + l_block[i];
                }


                // 3. calculate (P@V)[i,d] :
                // (P@V)[i, d] = sum_h(P[i,h]*V[h,d])
                // P is currently stored in S
                // cache friendliness: i -> h -> d:

                std::fill(PV.begin(), PV.end(), 0); // clear PV from previous block

                for (size_t i = 0; i < local_i_end; ++i) {
                    for (size_t h = 0; h < local_j_end; ++h) {
                        float p_value = S(0, i, h);
                        for (size_t d = 0; d < dim_v; ++d) {
                            PV(0, i, d) += p_value * V(b, j_block + h, d);
                        }
                    }
                }
                

                // 4. update output[i_block][0 : d] with new PV:
                // O[i_block][0 : d] = O[i_block][0 : d] * exp(m_acc - m_new) + PV[i_block][0 : d]
                for (size_t i = 0; i < local_i_end; ++i) {
                    float exp_coef = std::exp(m[i] - m_new[i]);
                    for (size_t d = 0; d < dim_v; ++d) {
                        output(b, i_block + i, d) = output(b, i_block + i, d) * exp_coef + PV(0, i, d);
                    }

                    m[i] = m_new[i];
                }
            }
            
            // 5. FlashAttention-2 rescaling:
            // after all j_blocks (all K and V)
            // and before O_block is changed:
            //
            // output is currently not divided by l[i] and it should be:
            // O[i_block] = O[i_block] / l[i_block]

            for (size_t i = i_block; i < i_end; ++i) {
                float scale = 1 / l[i - i_block];
                for (size_t d = 0; d < dim_v; ++d) {
                    output(b, i, d) = output(b, i, d) * scale;
                }
            }

            std::fill(m.begin(), m.end(), -std::numeric_limits<float>::infinity());

            // std::fill(m_new.begin(), m_new.end(), -std::numeric_limits<float>::infinity());

            std::fill(l.begin(), l.end(), 0);

        }
    }
    return output;
}


/*
Процесс вычисления Attention на xPU (подробно):

1. В кэш загружаются блоки матриц:
    Q[i_block : i_block + Br][0 : d]
    K[j_block : j_block + Bc][0 : d]
    V[j_block : j_block + Bc][0 : d]
    O[i_block : i_block + Br][0 : d] - матрица для записи результата.
2. Вычисляется блок матриц S и P:
    S = Q*K.T
    P = softmax(S)										- не материализуется в кэше отдельно, а перезаписывается поверх S.
    S[i_block : i_block + Br][j_block : j_block + Bc]	- находится в кэше.
    P[i_block : i_block + Br][j_block : j_block + Bc]	- находится в кэше.
    l_acc[i_block : i_block + Br]	- служебный вектор для текущих знаменателей строк матрицы P.
    m_acc[i_block : i_block + Br]	- служебный вектор для текущих максимумов строк матрицы P.
3. Обновляется блок матрицы O:
    O = P*V
    O_cur[i_block : i_block + Br][0 : d] - блок матрицы O, посчитанный для нового j_block'. Находится в кэше.
    O_acc[i_block : i_block + Br][0 : d] - блок матрицы O, аккумулированный со всех прошлых j_block.

    Корректный rescale при сложении блоков O_cur и O_acc:
    m_cur[i_block] = max_j(S[i_block][j_block])
    m_new[i_block] = max(m_cur, m_acc)
    P[i_block][j_block] = exp(S[i_block][j_block] - m_new[i_block])
    l_cur[i_block] = sum_j(P[i_block][j_block])
    l_new[i_block] = l_acc[i_block] * exp(m_acc - m_new) + l_cur[i_block]
    O_new[i_block] = (O_acc[i_block] * (l_acc[i_block] * exp(m_acc - m_new) + P[i_block][j_block] @ V[j_block]) / l_new[i_block]

4. Происходит смена j_block при фиксированном i_block.
    Q_block, O_block - фиксирован.
    K_block, V_block - сменяются.
5. Происходит смена i_block.
    Q_block и O_block полностью сменяются.
    Начинается расчет полностью новой области O_block.




Существует неоптимальность:
Мы много раз умножаем и делим на l_acc, хотя можем это сделать только в конце.
Данная версия используется в FlashAttention-2.

Обновим пункт 3:
3. Обновляется блок матрицы O:
    O = P*V
    O_cur[i_block : i_block + Br][0 : d] - блок матрицы O (неразделенной на l), посчитанный для нового j_block'. Находится в кэше.
    O_acc[i_block : i_block + Br][0 : d] - блок матрицы O (неразделенной на l), аккумулированный со всех прошлых j_block.

    Корректный rescale при сложении блоков O_cur и O_acc:
    m_cur[i_block] = max_j(S[i_block][j_block])
    m_new[i_block] = max(m_acc, m_cur)
    P[i_block][j_block] = exp(S[i_block][j_block] - m_new[i_block])
    l_cur[i_block] = sum_j(P_cur[i_block][j_block])
    l_new[i_block] = l_acc[i_block] * exp(m_acc - m_new) + l_cur[i_block]
    O_new[i_block] = O_acc[i_block] * exp(m_acc - m_new) + (P[i_block][j_block] @ V[j_block])

4. После всех блоков по j, перед изменением блока по i (изменение области записи O[i_block]):
    O[i_block] = O[i_block] / l_acc[i_block]
*/

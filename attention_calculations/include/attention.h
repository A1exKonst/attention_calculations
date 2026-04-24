#pragma once
#include "tensor.h"
#include "matmul_type.h"

Tensor attention_with_matmul(
	const Tensor& Q,
	const Tensor& K,
	const Tensor& V,
	MatMulType matmul_type
);

Tensor naive_attention(
	const Tensor& Q, 
	const Tensor& K, 
	const Tensor& V
);

Tensor cache_friendly_attention(
	const Tensor& Q,
	const Tensor& K,
	const Tensor& V
);

Tensor flash_attention(
	const Tensor& Q,
	const Tensor& K,
	const Tensor& V,
	const size_t block_size_rows = 32,
	const size_t block_size_columns = 32
);

Tensor tiled_attention(
	const Tensor& Q, 
	const Tensor& K, 
	const Tensor& V,
	uint64_t Bc = 32
);

Tensor vectorized_attention(
	const Tensor& Q, 
	const Tensor& K, 
	const Tensor& V,
	uint64_t Bc = 32
);
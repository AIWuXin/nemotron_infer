//
// Created by Administrator on 2026/6/15.
//

#ifndef NEMOTRON_INFER_EMBEDDING_CUH
#define NEMOTRON_INFER_EMBEDDING_CUH

#include <cuda_bf16.h>
#include "tensor/tensor.h"

namespace nemotron::ops {

// ===========================================================================
// Embedding 查表:  out[b,s,hidden] = table[token_id[b,s], :]
//
// 每个 token 在 embedding table 中查一行，copy 到输出对应位置。
// table [vocab_size, hidden] (BF16) + token_ids [B, S] (int64)
// → out [B, S, hidden] (BF16)
//
// 由于 vocab_size=131072 超大，token 通常 scatter 分布，
// 每次只访问 ~2048 行，L2 cache 对 table 覆盖差，瓶颈在显存。
// ===========================================================================

template<int Hidden>
__device__ __forceinline__ void embedding_lookup_kernel(
    const __nv_bfloat16* __restrict__ table,
    const int64_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    const size_t num_tokens  // B * S
) {
    for (size_t idx = blockIdx.x; idx < num_tokens; idx += gridDim.x) {
        const int64_t tok = token_ids[idx];
        const auto* src = reinterpret_cast<const float4*>(table + tok * Hidden);
        auto* dst = reinterpret_cast<float4*>(out + idx * Hidden);
        constexpr size_t vec_size = Hidden / 8;

        for (size_t v = threadIdx.x; v < vec_size; v += blockDim.x) {
            dst[v] = src[v];  // float4 copy = 8 个 BF16
        }
    }
}

template<int Hidden>
__global__ void embedding_lookup_launch(
    const __nv_bfloat16* __restrict__ table,
    const int64_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    const size_t num_tokens
) {
    embedding_lookup_kernel<Hidden>(table, token_ids, out, num_tokens);
}

template<int Hidden>
__host__ void embedding_lookup(
    const __nv_bfloat16* table,
    const int64_t* token_ids,
    __nv_bfloat16* out,
    const size_t num_tokens,
    cudaStream_t stream = nullptr
) {
    constexpr int block_dimx = 256;
    const int grid_dimx = min(num_tokens, static_cast<size_t>(65535));
    embedding_lookup_launch<Hidden><<<grid_dimx, block_dimx, 0, stream>>>(
        table, token_ids, out, num_tokens
    );
}

}  // namespace nemotron::ops

#endif //NEMOTRON_INFER_EMBEDDING_CUH

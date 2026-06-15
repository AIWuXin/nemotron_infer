//
// Created by Administrator on 2026/6/15.
//

#ifndef NEMOTRON_INFER_SDPA_PREFILL_CUH
#define NEMOTRON_INFER_SDPA_PREFILL_CUH

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

// cuDNN SDPA：USE_CUDNN 定义时走 cudnn_frontend graph (FlashAttention)，
// 否则用下方手写 FlashAttention。
// ⚠️ 只 include 声明头(.h)；frontend 实现在 sdpa_cudnn.cpp，由 MSVC 编译，
//    绝不能让 NVCC 处理 cudnn_frontend.h（cudafe++ 会崩溃）。
#ifdef USE_CUDNN
#include "ops/attention/sdpa_cudnn.h"
#endif

namespace nemotron::ops::attention {

// ===========================================================================
// 常量
// ===========================================================================
constexpr int SDPA_BC = 32;     // tile 大小
constexpr int SDPA_HEAD = 128;  // head_dim

// ===========================================================================
// warp 级 reduce（蝶式 shfl_xor：归约后所有 lane 持有相同结果）
//
// ⚠️ 必须用 __shfl_xor_sync 而非 __shfl_down_sync。
//    本 kernel 中每个 lane(=一个 KV 位置 tx) 都要用全局 m_new/d 去算
//    exp_score 与 rescale；shfl_down 只让 lane0 拿到正确归约值，其余 lane
//    是垃圾，会导致整个 softmax 错误（输出趋近 0）。
// ===========================================================================
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ===========================================================================
// 1. FP32 FlashAttention — 每 block 处理 1 head × 1 Q tile [Bc, head_dim]
//    blockDim = (32, 32), extern __shared__ smem[48 KB]
//    grid    = (num_heads, ceil(S/Bc))
// ===========================================================================
template<int Bc = SDPA_BC, int HEAD = SDPA_HEAD>
__device__ __forceinline__ void sdpa_prefill_kernel_fp32(
    const float* __restrict__ Q,    // [Bc, HEAD] per tile
    const float* __restrict__ K,    // [S, HEAD]  full seq
    const float* __restrict__ V,    // [S, HEAD]  full seq
    float* __restrict__ O,          // [Bc, HEAD] output for this tile
    const float rsqrt_d,
    const int S,
    float* __restrict__ Q_shared,
    float* __restrict__ K_shared,
    float* __restrict__ V_shared
) {
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // ---------- Q tile: global → shared (一次加载, 整轮复用) ----------
    int src_row = blockIdx.y * Bc + ty;  // 该 tile 对应的全局 Q 行
    {
        const float* Q_row = Q + src_row * HEAD;
        #pragma unroll
        for (int k = tx; k < HEAD; k += blockDim.x)
            Q_shared[ty * HEAD + k] = (src_row < S) ? Q_row[k] : 0.f;
    }
    __syncthreads();

    // 寄存器: 每线程持有输出 128/32=4 个元素
    float out[4] = {0.f, 0.f, 0.f, 0.f};
    float m_prev = -FLT_MAX;
    float d_prev = 0.f;

    // ---------- 流式扫 KV 块 ----------
    for (int tile_s = 0; tile_s < S; tile_s += Bc) {
        // --- load K tile ---
        {
            int kv_row = tile_s + ty;
            float* K_row = K_shared + ty * HEAD;
            #pragma unroll
            for (int k = tx; k < HEAD; k += blockDim.x)
                K_row[k] = (kv_row < S) ? K[kv_row * HEAD + k] : 0.f;
        }

        // --- load V tile ---
        {
            int kv_row = tile_s + ty;
            float* V_row = V_shared + ty * HEAD;
            #pragma unroll
            for (int k = tx; k < HEAD; k += blockDim.x)
                V_row[k] = (kv_row < S) ? V[kv_row * HEAD + k] : 0.f;
        }
        __syncthreads();

        // --- score = Q[ty] @ K[tx]^T / sqrt(d) ---
        const float* Q_row = Q_shared + ty * HEAD;
        const float* K_row = K_shared + tx * HEAD;
        float score = 0.f;
        #pragma unroll
        for (int k = 0; k < HEAD; k += 4) {
            float4 q = *(const float4*)(Q_row + k);
            float4 k_val = *(const float4*)(K_row + k);
            score += q.x * k_val.x + q.y * k_val.y + q.z * k_val.z + q.w * k_val.w;
        }
        score *= rsqrt_d;

        // --- causal mask: KV 位置 > Q 位置 → -inf ---
        if (tile_s + tx > src_row) score = -FLT_MAX;

        // --- online softmax ---
        float new_m = warp_reduce_max(score);
        float m_new = fmaxf(m_prev, new_m);
        float exp_score = expf(score - m_new);
        float new_d = warp_reduce_sum(exp_score);

        // 用 correction = exp(m_prev - m_new) 重缩放旧的(未归一化)输出。
        // 注意只乘 correction，不能乘 d_prev——out 累加的是 Σ exp·V，
        // 归一化留到最后 out /= d_prev。多 KV tile 时若混入 d_prev 会指数放大。
        float correction = expf(m_prev - m_new);
        #pragma unroll
        for (int k = 0; k < 4; k++) out[k] *= correction;
        d_prev = d_prev * correction + new_d;
        m_prev = m_new;

        // --- 累加新贡献: out[tx*4..] += Σ_s softmax[s] * V[s][tx*4..] ---
        #pragma unroll
        for (int s = 0; s < Bc; s++) {
            float sm = __shfl_sync(0xffffffff, exp_score, s);
            if (tile_s + s < S) {
                float4 v = *(const float4*)(V_shared + s * HEAD + tx * 4);
                out[0] += sm * v.x;
                out[1] += sm * v.y;
                out[2] += sm * v.z;
                out[3] += sm * v.w;
            }
        }

        __syncthreads();  // 下一轮 KV tile
    }

    // ---------- normalize & write ----------
    if (src_row < S) {
        #pragma unroll
        for (int k = 0; k < 4; k++) out[k] /= d_prev;
        float4* O_row = (float4*)(O + src_row * HEAD);
        O_row[tx] = make_float4(out[0], out[1], out[2], out[3]);
    }
}

// ===========================================================================
// __global__ launch fp32
// ===========================================================================
template<int Bc = SDPA_BC, int HEAD = SDPA_HEAD>
__global__ void sdpa_prefill_launch_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const float rsqrt_d,
    const int S,
    const int num_heads
) {
    extern __shared__ float smem[];
    float* Q_shared = smem;
    float* K_shared = smem + Bc * HEAD;
    float* V_shared = smem + 2 * Bc * HEAD;

    const int head = blockIdx.x;
    const size_t head_offset = (size_t)head * S * HEAD;

    sdpa_prefill_kernel_fp32<Bc, HEAD>(
        Q + head_offset, K + head_offset, V + head_offset,
        O + head_offset,
        rsqrt_d, S,
        Q_shared, K_shared, V_shared
    );
}

// ===========================================================================
// __host__ 调用入口
// ===========================================================================
template<int Bc = SDPA_BC, int HEAD = SDPA_HEAD>
__host__ void sdpa_prefill_fp32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int S,
    const int num_heads,
    cudaStream_t stream = nullptr
) {
    if (S == 0 || num_heads == 0) return;

    const float rsqrt_d = 1.f / sqrtf((float)HEAD);
    const dim3 block(Bc, Bc, 1);
    const dim3 grid(num_heads, (S + Bc - 1) / Bc, 1);
    const size_t smem_bytes = 3 * Bc * HEAD * sizeof(float);  // 48 KB

    sdpa_prefill_launch_fp32<Bc, HEAD><<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, rsqrt_d, S, num_heads
    );
}


// ===========================================================================
// 2. BF16 — 内部 FP32 计算，IO 类型为 BF16
//    USE_CUDNN 宏控制是否走 cuDNN SDPA
// ===========================================================================
template<int Bc = SDPA_BC, int HEAD = SDPA_HEAD>
__device__ __forceinline__ void sdpa_prefill_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const float rsqrt_d,
    const int S,
    float* __restrict__ Q_shared,
    float* __restrict__ K_shared,
    float* __restrict__ V_shared
) {
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // Q tile: bf16 → fp32
    int src_row = blockIdx.y * Bc + ty;
    {
        const __nv_bfloat16* Q_row = Q + src_row * HEAD;
        #pragma unroll
        for (int k = tx; k < HEAD; k += blockDim.x) {
            Q_shared[ty * HEAD + k] = (src_row < S)
                ? __bfloat162float(Q_row[k]) : 0.f;
        }
    }
    __syncthreads();

    float out[4] = {0.f, 0.f, 0.f, 0.f};
    float m_prev = -FLT_MAX;
    float d_prev = 0.f;

    for (int tile_s = 0; tile_s < S; tile_s += Bc) {
        // load K (bf16 → fp32)
        {
            int kv_row = tile_s + ty;
            float* K_row = K_shared + ty * HEAD;
            #pragma unroll
            for (int k = tx; k < HEAD; k += blockDim.x)
                K_row[k] = (kv_row < S) ? __bfloat162float(K[kv_row * HEAD + k]) : 0.f;
        }

        // load V (bf16 → fp32)
        {
            int kv_row = tile_s + ty;
            float* V_row = V_shared + ty * HEAD;
            #pragma unroll
            for (int k = tx; k < HEAD; k += blockDim.x)
                V_row[k] = (kv_row < S) ? __bfloat162float(V[kv_row * HEAD + k]) : 0.f;
        }
        __syncthreads();

        // score
        const float* Q_row = Q_shared + ty * HEAD;
        const float* K_row = K_shared + tx * HEAD;
        float score = 0.f;
        #pragma unroll
        for (int k = 0; k < HEAD; k += 4) {
            float4 q = *(const float4*)(Q_row + k);
            float4 k_val = *(const float4*)(K_row + k);
            score += q.x * k_val.x + q.y * k_val.y + q.z * k_val.z + q.w * k_val.w;
        }
        score *= rsqrt_d;

        if (tile_s + tx > src_row) score = -FLT_MAX;

        float new_m = warp_reduce_max(score);
        float m_new = fmaxf(m_prev, new_m);
        float exp_score = expf(score - m_new);
        float new_d = warp_reduce_sum(exp_score);

        float rescale = d_prev * expf(m_prev - m_new);
        #pragma unroll
        for (int k = 0; k < 4; k++) out[k] *= rescale;
        d_prev = d_prev * expf(m_prev - m_new) + new_d;
        m_prev = m_new;

        #pragma unroll
        for (int s = 0; s < Bc; s++) {
            float sm = __shfl_sync(0xffffffff, exp_score, s);
            if (tile_s + s < S) {
                float4 v = *(const float4*)(V_shared + s * HEAD + tx * 4);
                out[0] += sm * v.x;
                out[1] += sm * v.y;
                out[2] += sm * v.z;
                out[3] += sm * v.w;
            }
        }

        __syncthreads();
    }

    // write out: fp32 → bf16
    if (src_row < S) {
        #pragma unroll
        for (int k = 0; k < 4; k++) out[k] /= d_prev;
        __nv_bfloat16* O_row = O + src_row * HEAD;
        #pragma unroll
        for (int k = 0; k < 4; k++)
            O_row[tx * 4 + k] = __float2bfloat16_rn(out[k]);
    }
}

// ===========================================================================
// __global__ launch bf16
// ===========================================================================
template<int Bc = SDPA_BC, int HEAD = SDPA_HEAD>
__global__ void sdpa_prefill_launch_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const float rsqrt_d,
    const int S,
    const int num_heads
) {
    extern __shared__ float smem[];
    float* Q_shared = smem;
    float* K_shared = smem + Bc * HEAD;
    float* V_shared = smem + 2 * Bc * HEAD;

    const int head = blockIdx.x;
    const size_t head_offset = (size_t)head * S * HEAD;

    sdpa_prefill_kernel_bf16<Bc, HEAD>(
        Q + head_offset, K + head_offset, V + head_offset,
        O + head_offset,
        rsqrt_d, S,
        Q_shared, K_shared, V_shared
    );
}

// ===========================================================================
// __host__ 入口 — BF16，USE_CUDNN 宏控制
// ===========================================================================
template<int Bc = SDPA_BC, int HEAD = SDPA_HEAD>
__host__ void sdpa_prefill_bf16(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    const int S,
    const int num_heads,
    cudaStream_t stream = nullptr
) {
    if (S == 0 || num_heads == 0) return;

#ifdef USE_CUDNN
    // cuDNN frontend FlashAttention（成功则返回；失败回退手写）。
    if (sdpa_prefill_bf16_cudnn(Q, K, V, O, S, num_heads, HEAD, stream))
        return;
#endif

    // 手写 FlashAttention
    const float rsqrt_d = 1.f / sqrtf((float)HEAD);
    const dim3 block(Bc, Bc, 1);
    const dim3 grid(num_heads, (S + Bc - 1) / Bc, 1);
    const size_t smem_bytes = 3 * Bc * HEAD * sizeof(float);

    sdpa_prefill_launch_bf16<Bc, HEAD><<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, rsqrt_d, S, num_heads
    );
}

}  // namespace nemotron::ops::attention

#endif //NEMOTRON_INFER_SDPA_PREFILL_CUH

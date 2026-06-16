//
// Created by Administrator on 2026/6/16.
// Mamba-2 Causal Conv1D + SiLU
// ===========================================================================
// 深度可分卷积:  每组有独立的 conv1d 核 [4]
// 输入 x [B, S, conv_dim]      conv_dim=9728
// 权重 w [conv_dim, 1, 4]
// 偏置 b [conv_dim]
// 输出 y [B, S, conv_dim]      经过 silu(conv(x))
// ===========================================================================

#ifndef NEMOTRON_INFER_CAUSAL_CONV1D_CUH
#define NEMOTRON_INFER_CAUSAL_CONV1D_CUH

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

namespace nemotron::ops::mamba2 {

// ===========================================================================
// Prefill: 深度因果 Conv1D (kernel=4) + SiLU
//
// 时间维并行：kernel=4 的因果卷积 y[t] 只依赖 x[t..t-3]，没有长递归，
// 故沿 (channel, time-chunk) 二维展开。每个线程负责 1 个 channel 的
// TIME_CHUNK 个连续时间步，仅在 chunk 内串行（保留 3 元素寄存器滑窗、
// 每个 x 只读一次），chunk 之间完全并行 → 占用率拉满、访存延迟被盖住，
// kernel 进入带宽瓶颈区（此时 BF16 才能兑现 ~2x 带宽优势）。
//
// 线程映射（blockDim = (CH_TILE, TY)）：
//   ch    = blockIdx.x*CH_TILE + threadIdx.x   ← warp 内连续，coalesced
//   chunk = blockIdx.y*TY      + threadIdx.y   ← 覆盖 [t0, t0+TIME_CHUNK)
//   b_idx = blockIdx.z
// state_out 只由含 t=S-1 的 chunk 写出（每 channel 恰好一个线程）。
// ===========================================================================
template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__device__ __forceinline__ void causal_conv1d_prefill_kernel_fp32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ state_out,
    const int conv_dim,
    const int S,
    const int batch
) {
    const int ch = blockIdx.x * CH_TILE + threadIdx.x;
    if (ch >= conv_dim) return;
    const int chunk = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = chunk * TIME_CHUNK;
    if (t0 >= S) return;
    const int t1 = min(t0 + TIME_CHUNK, S);
    const int b_idx = blockIdx.z;

    const float* x_b = x + (size_t)b_idx * S * conv_dim;
    float* y_b = y + (size_t)b_idx * S * conv_dim;

    float weight[KERNEL];
    #pragma unroll
    for (int k = 0; k < KERNEL; ++k) weight[k] = w[ch * KERNEL + k];
    const float bias_v = b[ch];

    // 滑窗 state[j] = x[t0-(KERNEL-1)+j]（越界补 0），用于无依赖地启动本 chunk
    float state[KERNEL - 1];
    #pragma unroll
    for (int j = 0; j < KERNEL - 1; ++j) {
        const int ti = t0 - (KERNEL - 1) + j;
        state[j] = (ti >= 0) ? x_b[(size_t)ti * conv_dim + ch] : 0.f;
    }

    for (int t = t0; t < t1; ++t) {
        const float x_t = x_b[(size_t)t * conv_dim + ch];
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) acc += weight[k] * state[k];
        acc += weight[KERNEL - 1] * x_t;

        y_b[(size_t)t * conv_dim + ch] = acc / (1.f + expf(-acc));

        #pragma unroll
        for (int k = 0; k < KERNEL - 2; ++k) state[k] = state[k + 1];
        state[KERNEL - 2] = x_t;
    }

    // 仅末尾 chunk 落 decode 续接 state（此时 state = x[S-3..S-1]）
    if (t1 == S && state_out) {
        float* s = state_out + (size_t)b_idx * conv_dim * (KERNEL - 1)
                             + (size_t)ch * (KERNEL - 1);
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) s[k] = state[k];
    }
}

// ===========================================================================
// __global__ launch
// ===========================================================================
template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__global__ void causal_conv1d_prefill_launch_fp32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ state_out,
    const int conv_dim,
    const int S,
    const int batch
) {
    causal_conv1d_prefill_kernel_fp32<CH_TILE, TIME_CHUNK, KERNEL>(
        x, w, b, y, state_out, conv_dim, S, batch);
}

// ===========================================================================
// __host__ wrapper
// ===========================================================================
template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__host__ void causal_conv1d_prefill_fp32(
    const float* x, const float* w, const float* b, float* y,
    float* state_out, int conv_dim, int S, int batch,
    cudaStream_t stream = nullptr
) {
    constexpr int TY = 4;
    const int nchunks = (S + TIME_CHUNK - 1) / TIME_CHUNK;
    dim3 block(CH_TILE, TY);
    dim3 grid((conv_dim + CH_TILE - 1) / CH_TILE, (nchunks + TY - 1) / TY, batch);
    causal_conv1d_prefill_launch_fp32<CH_TILE, TIME_CHUNK, KERNEL><<<grid, block, 0, stream>>>(
        x, w, b, y, state_out, conv_dim, S, batch);
}

// ===========================================================================
// Decode: 单步 Conv1D + SiLU (基于缓存的 state)
// ===========================================================================
template<int KERNEL = 4>
__device__ __forceinline__ void causal_conv1d_decode_kernel_fp32(
    const float* __restrict__ x,     // [B, conv_dim] 当前 token
    const float* __restrict__ w,     // [conv_dim, KERNEL]
    const float* __restrict__ b,     // [conv_dim]
    float* __restrict__ y,           // [B, conv_dim]
    float* __restrict__ state,       // [B, conv_dim, KERNEL-1] in/out
    const int conv_dim,
    const int batch
) {
    const int tx = threadIdx.x;
    const int ch = blockIdx.x * blockDim.x + tx;
    if (ch >= conv_dim) return;

    const float* w_ch = w + ch * KERNEL;
    const float bias = b[ch];

    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        float* s = state + b_idx * conv_dim * (KERNEL - 1) + ch * (KERNEL - 1);
        float s_reg[KERNEL - 1];
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) s_reg[k] = s[k];

        const float x_t = x[b_idx * conv_dim + ch];

        float acc = bias;
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) acc += w_ch[k] * s_reg[k];
        acc += w_ch[KERNEL - 1] * x_t;

        const float silu_val = acc / (1.f + expf(-acc));
        y[b_idx * conv_dim + ch] = silu_val;

        // 更新 state
        #pragma unroll
        for (int k = 0; k < KERNEL - 2; ++k) s[k] = s_reg[k + 1];
        s[KERNEL - 2] = x_t;
    }
}

// ===========================================================================
// Decode __global__ + __host__
// ===========================================================================
template<int KERNEL = 4>
__global__ void causal_conv1d_decode_launch_fp32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ state,
    const int conv_dim,
    const int batch
) {
    causal_conv1d_decode_kernel_fp32<KERNEL>(x, w, b, y, state, conv_dim, batch);
}

template<int KERNEL = 4>
__host__ void causal_conv1d_decode_fp32(
    const float* x, const float* w, const float* b, float* y,
    float* state, int conv_dim, int batch,
    cudaStream_t stream = nullptr
) {
    const int threads = 128;
    const int grid = (conv_dim * batch + threads - 1) / threads;
    causal_conv1d_decode_launch_fp32<KERNEL><<<grid, threads, 0, stream>>>(
        x, w, b, y, state, conv_dim, batch);
}


// ===========================================================================
// BF16 版本
// ===========================================================================
template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__device__ __forceinline__ void causal_conv1d_prefill_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ state_out,
    const int conv_dim,
    const int S,
    const int batch
) {
    const int ch = blockIdx.x * CH_TILE + threadIdx.x;
    if (ch >= conv_dim) return;
    const int chunk = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = chunk * TIME_CHUNK;
    if (t0 >= S) return;
    const int t1 = min(t0 + TIME_CHUNK, S);
    const int b_idx = blockIdx.z;

    const __nv_bfloat16* x_b = x + (size_t)b_idx * S * conv_dim;
    __nv_bfloat16* y_b = y + (size_t)b_idx * S * conv_dim;

    float weight[KERNEL];
    #pragma unroll
    for (int k = 0; k < KERNEL; ++k) weight[k] = w[ch * KERNEL + k];
    const float bias_v = b[ch];

    float state[KERNEL - 1];
    #pragma unroll
    for (int j = 0; j < KERNEL - 1; ++j) {
        const int ti = t0 - (KERNEL - 1) + j;
        state[j] = (ti >= 0) ? __bfloat162float(x_b[(size_t)ti * conv_dim + ch]) : 0.f;
    }

    for (int t = t0; t < t1; ++t) {
        const float x_t = __bfloat162float(x_b[(size_t)t * conv_dim + ch]);
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) acc += weight[k] * state[k];
        acc += weight[KERNEL - 1] * x_t;

        y_b[(size_t)t * conv_dim + ch] = __float2bfloat16_rn(acc / (1.f + expf(-acc)));

        #pragma unroll
        for (int k = 0; k < KERNEL - 2; ++k) state[k] = state[k + 1];
        state[KERNEL - 2] = x_t;
    }

    if (t1 == S && state_out) {
        float* s = state_out + (size_t)b_idx * conv_dim * (KERNEL - 1)
                             + (size_t)ch * (KERNEL - 1);
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) s[k] = state[k];
    }
}

template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__global__ void causal_conv1d_prefill_launch_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ state_out,
    const int conv_dim,
    const int S,
    const int batch
) {
    causal_conv1d_prefill_kernel_bf16<CH_TILE, TIME_CHUNK, KERNEL>(
        x, w, b, y, state_out, conv_dim, S, batch);
}

// ---------------------------------------------------------------------------
// BF16 向量化路径：每线程处理 2 个相邻 channel，用 bf162 读/写。
// warp 内 32 线程覆盖 64 个 channel = 128B 连续事务（标量版每 warp 仅 64B，
// 半 cacheline，事务效率打折）。要求 conv_dim 为偶数（host wrapper 据此分发，
// 奇数回退标量 time-parallel 版）；conv_dim 偶数保证 ch 偶数起点、4B 对齐、
// 且 ch<conv_dim ⇒ ch+1 必有效，故内部无需逐元素边界判断。
// ---------------------------------------------------------------------------
template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__device__ __forceinline__ void causal_conv1d_prefill_kernel_bf16_vec2(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ state_out,
    const int conv_dim,
    const int S,
    const int batch
) {
    const int ch = (blockIdx.x * CH_TILE + threadIdx.x) * 2;
    if (ch >= conv_dim) return;
    const int chunk = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = chunk * TIME_CHUNK;
    if (t0 >= S) return;
    const int t1 = min(t0 + TIME_CHUNK, S);
    const int b_idx = blockIdx.z;

    const __nv_bfloat16* x_b = x + (size_t)b_idx * S * conv_dim;
    __nv_bfloat16* y_b = y + (size_t)b_idx * S * conv_dim;

    float wa[KERNEL], wb[KERNEL];
    #pragma unroll
    for (int k = 0; k < KERNEL; ++k) {
        wa[k] = w[ch * KERNEL + k];
        wb[k] = w[(ch + 1) * KERNEL + k];
    }
    const float ba = b[ch], bb = b[ch + 1];

    float sa[KERNEL - 1], sb[KERNEL - 1];
    #pragma unroll
    for (int j = 0; j < KERNEL - 1; ++j) {
        const int ti = t0 - (KERNEL - 1) + j;
        if (ti >= 0) {
            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(
                &x_b[(size_t)ti * conv_dim + ch]);
            sa[j] = __bfloat162float(v.x);
            sb[j] = __bfloat162float(v.y);
        } else { sa[j] = 0.f; sb[j] = 0.f; }
    }

    for (int t = t0; t < t1; ++t) {
        const __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(
            &x_b[(size_t)t * conv_dim + ch]);
        const float xa = __bfloat162float(xv.x);
        const float xb = __bfloat162float(xv.y);

        float acc_a = ba, acc_b = bb;
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) { acc_a += wa[k] * sa[k]; acc_b += wb[k] * sb[k]; }
        acc_a += wa[KERNEL - 1] * xa;
        acc_b += wb[KERNEL - 1] * xb;

        const __nv_bfloat162 yv = __floats2bfloat162_rn(
            acc_a / (1.f + expf(-acc_a)), acc_b / (1.f + expf(-acc_b)));
        *reinterpret_cast<__nv_bfloat162*>(&y_b[(size_t)t * conv_dim + ch]) = yv;

        #pragma unroll
        for (int k = 0; k < KERNEL - 2; ++k) { sa[k] = sa[k + 1]; sb[k] = sb[k + 1]; }
        sa[KERNEL - 2] = xa; sb[KERNEL - 2] = xb;
    }

    if (t1 == S && state_out) {
        float* s = state_out + (size_t)b_idx * conv_dim * (KERNEL - 1);
        float* sa_o = s + (size_t)ch * (KERNEL - 1);
        float* sb_o = s + (size_t)(ch + 1) * (KERNEL - 1);
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) { sa_o[k] = sa[k]; sb_o[k] = sb[k]; }
    }
}

template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__global__ void causal_conv1d_prefill_launch_bf16_vec2(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ state_out,
    const int conv_dim,
    const int S,
    const int batch
) {
    causal_conv1d_prefill_kernel_bf16_vec2<CH_TILE, TIME_CHUNK, KERNEL>(
        x, w, b, y, state_out, conv_dim, S, batch);
}

template<int CH_TILE = 64, int TIME_CHUNK = 64, int KERNEL = 4>
__host__ void causal_conv1d_prefill_bf16(
    const __nv_bfloat16* x, const float* w, const float* b,
    __nv_bfloat16* y, float* state_out,
    int conv_dim, int S, int batch,
    cudaStream_t stream = nullptr
) {
    constexpr int TY = 4;
    const int nchunks = (S + TIME_CHUNK - 1) / TIME_CHUNK;
    dim3 block(CH_TILE, TY);
    if ((conv_dim & 1) == 0) {
        // 向量化：grid.x 按 channel-pair 划分
        dim3 grid((conv_dim / 2 + CH_TILE - 1) / CH_TILE, (nchunks + TY - 1) / TY, batch);
        causal_conv1d_prefill_launch_bf16_vec2<CH_TILE, TIME_CHUNK, KERNEL><<<grid, block, 0, stream>>>(
            x, w, b, y, state_out, conv_dim, S, batch);
    } else {
        dim3 grid((conv_dim + CH_TILE - 1) / CH_TILE, (nchunks + TY - 1) / TY, batch);
        causal_conv1d_prefill_launch_bf16<CH_TILE, TIME_CHUNK, KERNEL><<<grid, block, 0, stream>>>(
            x, w, b, y, state_out, conv_dim, S, batch);
    }
}

// BF16 decode
template<int KERNEL = 4>
__device__ __forceinline__ void causal_conv1d_decode_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ state,
    const int conv_dim,
    const int batch
) {
    const int tx = threadIdx.x;
    const int ch = blockIdx.x * blockDim.x + tx;
    if (ch >= conv_dim) return;

    const float* w_ch = w + ch * KERNEL;
    const float bias_v = b[ch];

    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        float* s = state + b_idx * conv_dim * (KERNEL - 1) + ch * (KERNEL - 1);
        float s_reg[KERNEL - 1];
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) s_reg[k] = s[k];

        const float x_t = __bfloat162float(x[b_idx * conv_dim + ch]);

        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < KERNEL - 1; ++k) acc += w_ch[k] * s_reg[k];
        acc += w_ch[KERNEL - 1] * x_t;

        const float silu_val = acc / (1.f + expf(-acc));
        y[b_idx * conv_dim + ch] = __float2bfloat16_rn(silu_val);

        #pragma unroll
        for (int k = 0; k < KERNEL - 2; ++k) s[k] = s_reg[k + 1];
        s[KERNEL - 2] = x_t;
    }
}

template<int KERNEL = 4>
__global__ void causal_conv1d_decode_launch_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ state,
    const int conv_dim,
    const int batch
) {
    causal_conv1d_decode_kernel_bf16<KERNEL>(x, w, b, y, state, conv_dim, batch);
}

template<int KERNEL = 4>
__host__ void causal_conv1d_decode_bf16(
    const __nv_bfloat16* x, const float* w, const float* b,
    __nv_bfloat16* y, float* state,
    int conv_dim, int batch,
    cudaStream_t stream = nullptr
) {
    const int threads = 128;
    const int grid = (conv_dim * batch + threads - 1) / threads;
    causal_conv1d_decode_launch_bf16<KERNEL><<<grid, threads, 0, stream>>>(
        x, w, b, y, state, conv_dim, batch);
}

}  // namespace nemotron::ops::mamba2

#endif //NEMOTRON_INFER_CAUSAL_CONV1D_CUH

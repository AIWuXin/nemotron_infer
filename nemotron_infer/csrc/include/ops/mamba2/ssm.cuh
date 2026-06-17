//
// Created by Administrator on 2026/6/16.
// Mamba-2 SSM Decode — 单步循环状态更新
// ===========================================================================
// 布局：state [B, H, D, N]
//   H = mamba_num_heads = 96
//   D = head_dim = 80
//   N = ssm_state_size = 128
// ===========================================================================
#ifndef NEMOTRON_INFER_MAMBA2_SSM_CUH
#define NEMOTRON_INFER_MAMBA2_SSM_CUH

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

namespace nemotron::ops::mamba2 {

// I/O dtype 适配：x/dt/B/C/y 可为 fp32 或 bf16，state 恒为 fp32（递归累加器）
__device__ __forceinline__ float ssm_to_f(float v)          { return v; }
__device__ __forceinline__ float ssm_to_f(__nv_bfloat16 v)  { return __bfloat162float(v); }
__device__ __forceinline__ void  ssm_store(float* p, float v)         { *p = v; }
__device__ __forceinline__ void  ssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// ===========================================================================
// 合并访存的 SSM 单步 decode（fp32/bf16 共用一份实现）
//
// 关键优化 vs 朴素版（一线程一 pos、内层串行 n）：
//   - 朴素版 warp 内相邻线程访问的 state 地址差 N*4=512B → 32 条 cache line，完全未合并。
//   - 本版改为 **warp-per-pos**：32 个 lane 沿 N 维连续划分（n = lane, lane+32, ...），
//     相邻 lane 命中相邻 state 地址 → 合并事务；y 由 warp shfl 归约。
//   - B/C 一个 head 内被所有 pos 共享，先协作load到 shared memory，消除 D 倍冗余全局读。
// 布局：1 block = 1 head，blockDim=128（4 warp），每 warp 串行处理 pos=warpId, +nWarps,...
// state 恒 fp32：bf16 路径仅 x/dt/B/C/y 走 bf16，累加器精度不降。
//
// 注意：H/D/N 全部显式必填，不给默认值——循环次数与 state stride 依赖编译期 N，
// 漏传会静默用错 N 跑出"恒定比例偏大"的错误数值（越界读到相邻 arena）。
// ===========================================================================
template<typename IoT, int H, int D, int N>
__device__ __forceinline__ void ssm_decode_dev(
    const IoT* __restrict__ x,          // [B, H*D]
    const IoT* __restrict__ dt,         // [B, H]
    const float* __restrict__ A_log,    // [H]
    const IoT* __restrict__ B,          // [B, g, N]  g=n_groups
    const IoT* __restrict__ C,          // [B, g, N]
    const float* __restrict__ D_param,  // [H]
    float* __restrict__ state,          // [B, H, D, N]  恒 fp32
    IoT* __restrict__ y,                // [B, H*D]
    const float* __restrict__ dt_bias,  // [H]  optional
    const int batch, const int n_groups,
    const float dt_min, const float dt_max
) {
    const int head = blockIdx.x;
    if (head >= H * batch) return;
    const int b_idx = head / H;
    const int h = head % H;
    const int group = h / (H / n_groups);

    const int lane    = threadIdx.x & 31;
    const int warpId  = threadIdx.x >> 5;
    const int nWarps  = blockDim.x  >> 5;

    // B/C 缓存进 shared memory：一个 head 内所有 pos 复用同一组 [N]
    __shared__ float sB[N];
    __shared__ float sC[N];
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sB[i] = ssm_to_f(B[b_idx * n_groups * N + group * N + i]);
        sC[i] = ssm_to_f(C[b_idx * n_groups * N + group * N + i]);
    }
    __syncthreads();

    // dt 离散化 softplus+clamp（每线程算同一标量，便宜）
    // clamp 上下界由调用方传入：本模型 dt_limit=(0,inf) → (dt_min=0, dt_max=FLT_MAX)，
    // 即只 softplus 不截断；与 SSD scan prefill 保持一致（衔接同一条 state）。
    float dt_v = ssm_to_f(dt[b_idx * H + h]);
    if (dt_bias) dt_v += dt_bias[h];
    const float sp = (dt_v > 20.f) ? dt_v : (dt_v < -20.f) ? 0.f : logf(1.f + expf(dt_v));
    dt_v = fminf(dt_max, fmaxf(dt_min, sp));
    // 衰减因子 dA = exp(dt·A)，A=-exp(A_log)（A 不预乘 dt）→ exp(-exp(A_log)·dt)，线性于 dt。
    // 与 SSD scan prefill（已对齐 HF 1e-6）一致，衔接同一条 state。
    const float dA = expf(-expf(A_log[h]) * dt_v);
    const float Dh = D_param[h];

    const size_t head_base = (size_t)b_idx * H * D * N + (size_t)h * D * N;

    for (int pos = warpId; pos < D; pos += nWarps) {
        const float x_val = ssm_to_f(x[b_idx * H * D + h * D + pos]);
        float* s_pos = state + head_base + (size_t)pos * N;

        float y_part = 0.f;
        for (int n = lane; n < N; n += 32) {           // 合并：相邻 lane → 相邻 n
            const float s_new = s_pos[n] * dA + dt_v * sB[n] * x_val;
            s_pos[n] = s_new;
            y_part += s_new * sC[n];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)         // warp 归约求和 → lane 0
            y_part += __shfl_down_sync(0xffffffff, y_part, off);

        if (lane == 0)
            ssm_store(&y[b_idx * H * D + h * D + pos], y_part + Dh * x_val);
    }
}

template<int H, int D, int N>
__global__ void ssm_decode_fp32(
    const float* __restrict__ x, const float* __restrict__ dt,
    const float* __restrict__ A_log, const float* __restrict__ B,
    const float* __restrict__ C, const float* __restrict__ D_param,
    float* __restrict__ state, float* __restrict__ y,
    const float* __restrict__ dt_bias, const int batch, const int n_groups,
    const float dt_min = 0.f, const float dt_max = FLT_MAX
) {
    ssm_decode_dev<float, H, D, N>(x, dt, A_log, B, C, D_param, state, y,
                                   dt_bias, batch, n_groups, dt_min, dt_max);
}

template<int H, int D, int N>
__global__ void ssm_decode_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dt,
    const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C, const float* __restrict__ D_param,
    float* __restrict__ state, __nv_bfloat16* __restrict__ y,
    const float* __restrict__ dt_bias, const int batch, const int n_groups,
    const float dt_min = 0.f, const float dt_max = FLT_MAX
) {
    ssm_decode_dev<__nv_bfloat16, H, D, N>(x, dt, A_log, B, C, D_param, state, y,
                                           dt_bias, batch, n_groups, dt_min, dt_max);
}

}  // namespace nemotron::ops::mamba2

#endif

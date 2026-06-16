//
// Mamba-2 SSD scan — prefill（递归等价形式，FP32）
// ===========================================================================
// HF 的 chunked scan（分块 + 下三角矩阵 + 两次 segment_sum）在数学上等价于
// 朴素的顺序 SSM 递归；分块只是为了上 tensor core。Phase 1「手写跑通正确性」
// 这里实现递归形式（精度表要求该算子全程 FP32，BF16 会 NaN/掉精度）。
//
// 离散化与 HF modeling_nemotron_h.torch_forward 一致：
//   dt   = clamp(softplus(dt_raw + dt_bias), dt_min, dt_max)   // 本模型 dt_limit=(0,inf)
//   A_d  = -exp(A_log) * dt   →  每步衰减 dA = exp(A_d)
//   x_d  = x * dt             →  state[p,n] = state[p,n]*dA + (dt*x[p])*B[n]
//   y[p] = Σ_n state[p,n]*C[n] + D*x_raw[p]                    // D skip 用未离散化 x
//
// 布局（与 HF 一致，B=batch）：
//   x,y          [B, S, H, D]   D=head_dim
//   dt           [B, S, H]
//   A_log,D,bias [H]
//   B,C          [B, S, G, N]   G=n_groups（每 head 内 D 个 pos 共享同一 group）
//   ssm_state    [B, H, D, N]   可选，prefill 末态供 decode 续接
//
// 并行：1 block = 同一 head 的 WARPS 个 pos；warp-per-pos，32 lane 沿 N 维连续
//   划分（合并访存），state 留寄存器跨 S 步；B/C 每步载入 shared memory 复用。
//   grid=(B*H, ceil(D/WARPS))，block=(32, WARPS)。
// ===========================================================================
#ifndef NEMOTRON_INFER_MAMBA2_SSD_SCAN_CUH
#define NEMOTRON_INFER_MAMBA2_SSD_SCAN_CUH

#include <cfloat>
#include <cmath>

namespace nemotron::ops::mamba2 {

__device__ __forceinline__ float ssd_softplus_clamp(float v, float dt_min, float dt_max) {
    const float sp = (v > 20.f) ? v : (v < -20.f) ? 0.f : logf(1.f + expf(v));
    return fminf(dt_max, fmaxf(dt_min, sp));
}

template<int H, int D, int N, int WARPS = 4>
__global__ void ssd_scan_prefill_launch_fp32(
    const float* __restrict__ x,        // [B,S,H,D]
    const float* __restrict__ dt,       // [B,S,H]
    const float* __restrict__ A_log,    // [H]
    const float* __restrict__ B,        // [B,S,G,N]
    const float* __restrict__ C,        // [B,S,G,N]
    const float* __restrict__ D_param,  // [H]
    const float* __restrict__ dt_bias,  // [H] optional
    float* __restrict__ y,              // [B,S,H,D]
    float* __restrict__ ssm_state_out,  // [B,H,D,N] optional
    const int batch, const int S, const int n_groups,
    const float dt_min, const float dt_max
) {
    constexpr int NPL = (N + 31) / 32;          // 每 lane 持有的 state 数

    const int head = blockIdx.x;
    if (head >= H * batch) return;
    const int b_idx = head / H;
    const int h     = head % H;
    const int group = h / (H / n_groups);

    const int lane = threadIdx.x;               // 0..31
    const int warp = threadIdx.y;               // 0..WARPS-1
    const int pos  = blockIdx.y * WARPS + warp; // head_dim 内的位置
    const bool active = (pos < D);

    const float A_h    = -expf(A_log[h]);
    const float Dh     = D_param[h];
    const float bias_h = dt_bias ? dt_bias[h] : 0.f;

    __shared__ float sB[N];
    __shared__ float sC[N];
    const int tid = warp * 32 + lane;
    const int blockThreads = WARPS * 32;

    float st[NPL];
    #pragma unroll
    for (int k = 0; k < NPL; ++k) st[k] = 0.f;

    for (int t = 0; t < S; ++t) {
        // B/C[t]（按 group）载入 shared，供本 head 全部 pos 复用
        const size_t bc_base = ((size_t)(b_idx * S + t) * n_groups + group) * N;
        for (int i = tid; i < N; i += blockThreads) {
            sB[i] = B[bc_base + i];
            sC[i] = C[bc_base + i];
        }
        __syncthreads();

        if (active) {
            const float dt_v = ssd_softplus_clamp(
                dt[(size_t)(b_idx * S + t) * H + h] + bias_h, dt_min, dt_max);
            const float dA = expf(A_h * dt_v);
            const float x_val = x[((size_t)(b_idx * S + t) * H + h) * D + pos];

            float y_part = 0.f;
            #pragma unroll
            for (int k = 0; k < NPL; ++k) {
                const int n = lane + k * 32;
                if (n < N) {
                    st[k] = st[k] * dA + dt_v * sB[n] * x_val;   // 每个 (p,n) 独立递归
                    y_part += st[k] * sC[n];
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)               // warp 归约 → lane 0
                y_part += __shfl_down_sync(0xffffffff, y_part, off);

            if (lane == 0)
                y[((size_t)(b_idx * S + t) * H + h) * D + pos] = y_part + Dh * x_val;
        }
        __syncthreads();   // 下一步覆盖 shared 前同步（idle warp 也必须到达）
    }

    if (ssm_state_out && active) {
        float* s_out = ssm_state_out + ((size_t)(b_idx * H + h) * D + pos) * N;
        #pragma unroll
        for (int k = 0; k < NPL; ++k) {
            const int n = lane + k * 32;
            if (n < N) s_out[n] = st[k];
        }
    }
}

// ===========================================================================
// __host__ wrapper
// ===========================================================================
template<int H, int D, int N, int WARPS = 4>
__host__ void ssd_scan_prefill_fp32(
    const float* x, const float* dt, const float* A_log,
    const float* B, const float* C, const float* D_param, const float* dt_bias,
    float* y, float* ssm_state_out,
    int batch, int S, int n_groups,
    float dt_min = 0.f, float dt_max = FLT_MAX,
    cudaStream_t stream = nullptr
) {
    dim3 block(32, WARPS);
    dim3 grid(batch * H, (D + WARPS - 1) / WARPS);
    ssd_scan_prefill_launch_fp32<H, D, N, WARPS><<<grid, block, 0, stream>>>(
        x, dt, A_log, B, C, D_param, dt_bias, y, ssm_state_out,
        batch, S, n_groups, dt_min, dt_max);
}

}  // namespace nemotron::ops::mamba2

#endif  // NEMOTRON_INFER_MAMBA2_SSD_SCAN_CUH

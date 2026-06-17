//
// Mamba-2 SSD scan — prefill（chunked 形式，TF32 tensor core）
// ===========================================================================
// 数学等价于 ssd_scan.cuh 的顺序递归，但把重活重组成一组 batched GEMM（吃满
// tensor core），消除 O(S) 串行依赖。HF chunked SSD 的等价实现（见
// tools/dump_mamba_block.py 行 134-186），inter-chunk 用显式转移矩阵 T 表达
// （比 HF 的 transpose 体操更直观，数学相同）。
//
// 精度：cumsum / exp / segsum / decay 全程 fp32（数值红线）；G/M/Y 几个矩阵乘
//   走 TF32 tensor core（gemm_strided_batched_tf32）。IO 可 fp32 或 bf16，内部
//   workspace 恒 fp32。
//
// 记号：B batch，H heads，P head_dim，N state，C chunk，nc=ceil(S/C)，Sp=nc*C。
// 流程（每 (b,h)）：
//   离散化 x_d=x·dt[.,C,P]、Ad=-exp(A_log)·dt[.,C]、Bf/Cf[.,C,N]（repeat_interleave）
//   A_cumsum=cumsum(Ad)
//   G=Cf·Bfᵀ[C,C]；M=G·exp(Acs[t]-Acs[s])(s≤t)；Y_diag=M·x_d[C,P]
//   chunk_states=x_dᵀ·(Bf·decay_states)[P,N]
//   inter-chunk：T[z,c]=exp(PA[z]-PA[c+1])(c<z)，S_in=T·chunk_states；末态=Tlast·chunk_states
//   Y_off=(Cf·S_inᵀ)·exp(Acs)[C,P]；Y=Y_diag+Y_off+D·x_raw
// ===========================================================================
#ifndef NEMOTRON_INFER_MAMBA2_SSD_SCAN_CHUNKED_CUH
#define NEMOTRON_INFER_MAMBA2_SSD_SCAN_CHUNKED_CUH

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

#include "ops/gemm.cuh"
#include "ops/mamba2/ssd_scan.cuh"   // ssd_softplus_clamp, scan_to_f, scan_store

namespace nemotron::ops::mamba2 {

// ---------------------------------------------------------------------------
// K1: 离散化 + 重排到 [B,H,nc,C,*]（pad 末 chunk 补 0）
//   x_d = x·dt, Bf/Cf = B/C（per-head group）, Ad = -exp(A_log)·dt
// ---------------------------------------------------------------------------
template<typename IoT, int H, int P, int N, int C>
__global__ void ssdc_discretize(
    const IoT* __restrict__ x,        // [B,S,H,P]
    const IoT* __restrict__ dt,       // [B,S,H]
    const IoT* __restrict__ Bin,      // [B,S,G,N]
    const IoT* __restrict__ Cin,      // [B,S,G,N]
    const float* __restrict__ A_log,  // [H]
    const float* __restrict__ dt_bias,// [H]
    float* __restrict__ x_d,          // [B,H,nc,C,P]
    float* __restrict__ Bf,           // [B,H,nc,C,N]
    float* __restrict__ Cf,           // [B,H,nc,C,N]
    float* __restrict__ Ad,           // [B,H,nc,C]
    int batch, int S, int nc, int n_groups,
    float dt_min, float dt_max
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // over B*H*nc*C
    const long total = (long)batch * H * nc * C;
    if (idx >= total) return;

    const int t = idx % C;
    const int c = (idx / C) % nc;
    const int h = (idx / ((long)C * nc)) % H;
    const int b = idx / ((long)C * nc * H);
    const int s = c * C + t;
    const bool valid = (s < S);
    const int group = h / (H / n_groups);

    float dt_v = 0.f;
    if (valid) {
        const float bias = dt_bias ? dt_bias[h] : 0.f;
        dt_v = ssd_softplus_clamp(scan_to_f(dt[(long)(b * S + s) * H + h])
                                  + bias, dt_min, dt_max);
    }
    Ad[idx] = -expf(A_log[h]) * dt_v;

    const long xd_base = idx * P;
    const long xin_base = (long)(b * S + s) * H * P + (long)h * P;
    #pragma unroll
    for (int p = 0; p < P; ++p)
        x_d[xd_base + p] = valid ? scan_to_f(x[xin_base + p]) * dt_v : 0.f;

    const long bc_base = idx * N;
    const long in_base = (long)(b * S + s) * n_groups * N + (long)group * N;
    for (int n = 0; n < N; ++n) {
        Bf[bc_base + n] = valid ? scan_to_f(Bin[in_base + n]) : 0.f;
        Cf[bc_base + n] = valid ? scan_to_f(Cin[in_base + n]) : 0.f;
    }
}

// ---------------------------------------------------------------------------
// K2: chunk 内 cumsum（每 (b,h,c) 一个 block，lane0 串行 C 步），并存 Alast
// ---------------------------------------------------------------------------
template<int C>
__global__ void ssdc_cumsum(
    const float* __restrict__ Ad,     // [B,H,nc,C]
    float* __restrict__ Acs,          // [B,H,nc,C]
    float* __restrict__ Alast,        // [B,H,nc]
    int total_bhc                     // B*H*nc
) {
    const int bhc = blockIdx.x;
    if (bhc >= total_bhc) return;
    if (threadIdx.x != 0) return;
    const long base = (long)bhc * C;
    float run = 0.f;
    #pragma unroll
    for (int t = 0; t < C; ++t) { run += Ad[base + t]; Acs[base + t] = run; }
    Alast[bhc] = run;
}

// ---------------------------------------------------------------------------
// K3: M = G ⊙ L，L[t,s]=exp(Acs[t]-Acs[s]) (s≤t 否则 0)，in-place 到 GM
// ---------------------------------------------------------------------------
template<int C>
__global__ void ssdc_apply_decay_M(
    float* __restrict__ GM,           // [B,H,nc,C,C] (in: G, out: M)
    const float* __restrict__ Acs,    // [B,H,nc,C]
    long total_bhc                    // B*H*nc
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // over total_bhc*C*C
    if (idx >= total_bhc * (long)C * C) return;
    const int s = idx % C;
    const int t = (idx / C) % C;
    const long bhc = idx / ((long)C * C);
    if (s > t) { GM[idx] = 0.f; return; }
    const float* acs = Acs + bhc * C;
    GM[idx] *= expf(acs[t] - acs[s]);
}

// ---------------------------------------------------------------------------
// K4: tmp = Bf ⊙ decay_states，decay_states[t]=exp(Acs[C-1]-Acs[t])
// ---------------------------------------------------------------------------
template<int C, int N>
__global__ void ssdc_scale_B_decay(
    const float* __restrict__ Bf,     // [B,H,nc,C,N]
    const float* __restrict__ Acs,    // [B,H,nc,C]
    float* __restrict__ tmp,          // [B,H,nc,C,N]
    long total_bhc
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // over total_bhc*C*N
    if (idx >= total_bhc * (long)C * N) return;
    const int t = (idx / N) % C;
    const long bhc = idx / ((long)C * N);
    const float* acs = Acs + bhc * C;
    tmp[idx] = Bf[idx] * expf(acs[C - 1] - acs[t]);
}

// ---------------------------------------------------------------------------
// K5a: 每 (b,h) 前缀和 PA[nc+1]（PA[0]=0, PA[z]=PA[z-1]+Alast[z-1]）
// K5b: T[z,c]=exp(PA[z]-PA[c+1]) (c<z 否则 0)；Tlast[c]=exp(PA[nc]-PA[c+1])
// ---------------------------------------------------------------------------
__global__ void ssdc_prefix_A(
    const float* __restrict__ Alast,  // [B,H,nc]
    float* __restrict__ PA,           // [B,H,nc+1]
    int total_bh, int nc
) {
    const int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= total_bh) return;
    const float* al = Alast + (long)bh * nc;
    float* pa = PA + (long)bh * (nc + 1);
    float run = 0.f; pa[0] = 0.f;
    for (int z = 0; z < nc; ++z) { run += al[z]; pa[z + 1] = run; }
}

__global__ void ssdc_build_T(
    const float* __restrict__ PA,     // [B,H,nc+1]
    float* __restrict__ T,            // [B,H,nc,nc]
    float* __restrict__ Tlast,        // [B,H,nc]
    long total_bh, int nc
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // over total_bh*nc*nc
    if (idx >= total_bh * (long)nc * nc) return;
    const int c = idx % nc;
    const int z = (idx / nc) % nc;
    const long bh = idx / ((long)nc * nc);
    const float* pa = PA + bh * (nc + 1);
    T[idx] = (c < z) ? expf(pa[z] - pa[c + 1]) : 0.f;
    if (z == 0) Tlast[bh * nc + c] = expf(pa[nc] - pa[c + 1]);  // 每行各填一个，覆盖全 c
}

// ---------------------------------------------------------------------------
// K6: 组装 Y = Y_diag + (CS ⊙ exp(Acs)) + D·x_raw，写 IoT y[B,S,H,P]（unpad）
// ---------------------------------------------------------------------------
template<typename IoT, int H, int P, int N, int C>
__global__ void ssdc_finalize(
    const float* __restrict__ Ydiag,  // [B,H,nc,C,P]
    const float* __restrict__ CS,     // [B,H,nc,C,P]
    const float* __restrict__ Acs,    // [B,H,nc,C]
    const float* __restrict__ D_param,// [H]
    const IoT* __restrict__ x,        // [B,S,H,P] (x_raw, post-conv)
    IoT* __restrict__ y,              // [B,S,H,P]
    int batch, int S, int nc
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // over B*H*nc*C*P
    const long total = (long)batch * H * nc * C * P;
    if (idx >= total) return;
    const int p = idx % P;
    const int t = (idx / P) % C;
    const int c = (idx / ((long)P * C)) % nc;
    const int h = (idx / ((long)P * C * nc)) % H;
    const int b = idx / ((long)P * C * nc * H);
    const int s = c * C + t;
    if (s >= S) return;

    const long bhc = ((long)(b * H + h) * nc + c);
    const float decay = expf(Acs[bhc * C + t]);
    const float x_raw = scan_to_f(x[(long)(b * S + s) * H * P + (long)h * P + p]);
    const float val = Ydiag[idx] + CS[idx] * decay + D_param[h] * x_raw;
    scan_store(&y[(long)(b * S + s) * H * P + (long)h * P + p], val);
}

// ===========================================================================
// host wrapper
// ===========================================================================
template<typename IoT, int H, int P, int N, int C>
inline void ssd_scan_chunked_prefill(
    const IoT* x, const IoT* dt, const float* A_log,
    const IoT* B, const IoT* Cmat, const float* D_param, const float* dt_bias,
    IoT* y, float* ssm_state_out,
    int batch, int S, int n_groups,
    float dt_min = 0.f, float dt_max = FLT_MAX,
    cudaStream_t stream = nullptr
) {
    const int nc = (S + C - 1) / C;
    const long BH = (long)batch * H;
    const long BHC = BH * nc;

    auto wf = [](long n) { return allocate_tensor<float>(TensorShape::make_1d(n)); };
    auto x_d   = wf(BHC * C * P);
    auto Bf    = wf(BHC * C * N);
    auto Cf    = wf(BHC * C * N);
    auto Ad    = wf(BHC * C);
    auto Acs   = wf(BHC * C);
    auto Alast = wf(BHC);
    auto GM    = wf(BHC * C * C);
    auto Ydiag = wf(BHC * C * P);
    auto Bdec  = wf(BHC * C * N);
    auto cstate= wf(BHC * P * N);            // chunk_states [B,H,nc,P,N]
    auto PA    = wf(BH * (nc + 1));
    auto Tm    = wf(BH * nc * nc);
    auto Tlast = wf(BH * nc);
    auto S_in  = wf(BHC * P * N);            // [B,H,nc,P,N]
    auto CS    = wf(BHC * C * P);

    const int TPB = 256;
    auto ngrid = [&](long n) { return (int)((n + TPB - 1) / TPB); };

    // K1 离散化
    ssdc_discretize<IoT, H, P, N, C><<<ngrid(BHC * C), TPB, 0, stream>>>(
        x, dt, B, Cmat, A_log, dt_bias,
        x_d.data_, Bf.data_, Cf.data_, Ad.data_, batch, S, nc, n_groups, dt_min, dt_max);
    // K2 cumsum
    ssdc_cumsum<C><<<(int)BHC, 32, 0, stream>>>(Ad.data_, Acs.data_, Alast.data_, (int)BHC);

    // GEMM1: G = Cf @ Bfᵀ  [C,C]，batch BHC
    gemm_strided_batched_tf32(Cf.data_, Bf.data_, GM.data_,
        C, C, N, /*transA*/false, /*transB*/true,
        (long long)C * N, (long long)C * N, (long long)C * C, (int)BHC, stream);
    // K3 M = G ⊙ L
    ssdc_apply_decay_M<C><<<ngrid(BHC * C * C), TPB, 0, stream>>>(GM.data_, Acs.data_, BHC);
    // GEMM2: Y_diag = M @ x_d  [C,P]
    gemm_strided_batched_tf32(GM.data_, x_d.data_, Ydiag.data_,
        C, P, C, false, false,
        (long long)C * C, (long long)C * P, (long long)C * P, (int)BHC, stream);

    // K4 tmp = Bf ⊙ decay_states
    ssdc_scale_B_decay<C, N><<<ngrid(BHC * C * N), TPB, 0, stream>>>(
        Bf.data_, Acs.data_, Bdec.data_, BHC);
    // GEMM3: chunk_states = x_dᵀ @ tmp  [P,N]
    gemm_strided_batched_tf32(x_d.data_, Bdec.data_, cstate.data_,
        P, N, C, /*transA*/true, false,
        (long long)C * P, (long long)C * N, (long long)P * N, (int)BHC, stream);

    // K5 inter-chunk transition T
    ssdc_prefix_A<<<ngrid(BH), TPB, 0, stream>>>(Alast.data_, PA.data_, (int)BH, nc);
    ssdc_build_T<<<ngrid(BH * (long)nc * nc), TPB, 0, stream>>>(
        PA.data_, Tm.data_, Tlast.data_, BH, nc);
    // GEMM4: S_in = T @ chunk_states  [nc, P*N]，batch BH
    gemm_strided_batched_tf32(Tm.data_, cstate.data_, S_in.data_,
        nc, P * N, nc, false, false,
        (long long)nc * nc, (long long)nc * P * N, (long long)nc * P * N, (int)BH, stream);
    // GEMM5: ssm_state = Tlast @ chunk_states  [1, P*N]，batch BH（可选）
    if (ssm_state_out) {
        gemm_strided_batched_tf32(Tlast.data_, cstate.data_, ssm_state_out,
            1, P * N, nc, false, false,
            (long long)nc, (long long)nc * P * N, (long long)P * N, (int)BH, stream);
    }

    // GEMM6: CS = Cf @ S_inᵀ  [C,P]，batch BHC（S_in 每 chunk 一份 [P,N]）
    gemm_strided_batched_tf32(Cf.data_, S_in.data_, CS.data_,
        C, P, N, false, /*transB*/true,
        (long long)C * N, (long long)P * N, (long long)C * P, (int)BHC, stream);

    // K6 finalize
    ssdc_finalize<IoT, H, P, N, C><<<ngrid(BHC * C * P), TPB, 0, stream>>>(
        Ydiag.data_, CS.data_, Acs.data_, D_param, x, y, batch, S, nc);
}

// 便捷别名
template<int H, int P, int N, int C>
inline void ssd_scan_chunked_prefill_fp32(
    const float* x, const float* dt, const float* A_log,
    const float* B, const float* C_, const float* D_param, const float* dt_bias,
    float* y, float* ssm_state_out, int batch, int S, int n_groups,
    float dt_min = 0.f, float dt_max = FLT_MAX, cudaStream_t stream = nullptr
) {
    ssd_scan_chunked_prefill<float, H, P, N, C>(
        x, dt, A_log, B, C_, D_param, dt_bias, y, ssm_state_out,
        batch, S, n_groups, dt_min, dt_max, stream);
}

template<int H, int P, int N, int C>
inline void ssd_scan_chunked_prefill_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* dt, const float* A_log,
    const __nv_bfloat16* B, const __nv_bfloat16* C_, const float* D_param, const float* dt_bias,
    __nv_bfloat16* y, float* ssm_state_out, int batch, int S, int n_groups,
    float dt_min = 0.f, float dt_max = FLT_MAX, cudaStream_t stream = nullptr
) {
    ssd_scan_chunked_prefill<__nv_bfloat16, H, P, N, C>(
        x, dt, A_log, B, C_, D_param, dt_bias, y, ssm_state_out,
        batch, S, n_groups, dt_min, dt_max, stream);
}

}  // namespace nemotron::ops::mamba2

#endif  // NEMOTRON_INFER_MAMBA2_SSD_SCAN_CHUNKED_CUH

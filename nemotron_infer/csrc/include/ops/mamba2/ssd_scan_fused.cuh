//
// Mamba-2 SSD scan — prefill（fused / flash 形式）
// ===========================================================================
// 物化版（ssd_scan_chunked.cuh）撞带宽墙：把 G/M/x_d/Y 等 ~2.5GB 中间量写全局，
// 4060 仅 272GB/s → memory-bound，只 1.2x。本 fused 版把每 chunk 的 intra-chunk
// 重活（M=C·Bᵀ⊙decay、Y_diag=M·x_d、chunk_states）**全留 shared memory**，不写全局，
// 只物化小的 chunk_states[B,H,nc,P,N]。inter-chunk 递归复用 chunked.cuh 的 T 矩阵 gemm。
//
// 三趟：
//   pass1（每 (b,h,chunk) 一 block）：sB/sC/sX/sM 进 shared，算 Y_diag+D·x_raw 写 y、
//     chunk_states 写全局、Acs/Alast 写全局。
//   inter-chunk（复用）：prefix → build_T → S_in = T·chunk_states（+ ssm_state=Tlast·cs）。
//   pass3（每 (b,h,chunk) 一 block）：sC/S_in 进 shared，算 Y_off=(C·S_inᵀ)·exp(Acs)，y += Y_off。
//
// 精度：cumsum/exp/decay 与累加全 fp32（红线）；IO 可 fp32/bf16。C=32（shared ~46KB，
//   免 opt-in，占用率好）。
// ===========================================================================
#ifndef NEMOTRON_INFER_MAMBA2_SSD_SCAN_FUSED_CUH
#define NEMOTRON_INFER_MAMBA2_SSD_SCAN_FUSED_CUH

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

#include "ops/gemm.cuh"
#include "ops/mamba2/ssd_scan.cuh"          // ssd_softplus_clamp, scan_to_f/store
#include "ops/mamba2/ssd_scan_chunked.cuh"  // ssdc_prefix_A, ssdc_build_T

namespace nemotron::ops::mamba2 {

// ---------------------------------------------------------------------------
// pass1：intra-chunk 全 shared。grid=(B*H*nc)，block=256。
// ---------------------------------------------------------------------------
template<typename IoT, int H, int P, int N, int C>
__global__ void ssd_fused_pass1(
    const IoT* __restrict__ x,        // [B,S,H,P]
    const IoT* __restrict__ dt,       // [B,S,H]
    const IoT* __restrict__ Bin,      // [B,S,G,N]
    const IoT* __restrict__ Cin,      // [B,S,G,N]
    const float* __restrict__ A_log,  // [H]
    const float* __restrict__ D_param,// [H]
    const float* __restrict__ dt_bias,// [H]
    IoT* __restrict__ y,              // [B,S,H,P]  out: Y_diag + D·x_raw
    float* __restrict__ chunk_states, // [B,H,nc,P,N]
    float* __restrict__ Acs_g,        // [B,H,nc,C]
    float* __restrict__ Alast_g,      // [B,H,nc]
    int batch, int S, int nc, int n_groups, float dt_min, float dt_max
) {
    __shared__ float sB[C * N];
    __shared__ float sC[C * N];
    __shared__ float sX[C * P];   // x·dt
    __shared__ float sM[C * C];
    __shared__ float sAcs[C];
    __shared__ float sDt[C];
    __shared__ float sDecay[C];   // exp(Acs[C-1]-Acs[t])，chunk_states 复用

    const int bhc = blockIdx.x;
    const int c = bhc % nc;
    const int h = (bhc / nc) % H;
    const int b = bhc / (nc * H);
    const int group = h / (H / n_groups);
    const int base_s = c * C;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    const float A_h = -expf(A_log[h]);
    const float bias_h = dt_bias ? dt_bias[h] : 0.f;
    const float Dh = D_param[h];

    // dt / Ad
    for (int t = tid; t < C; t += nth) {
        const int s = base_s + t;
        float dtv = 0.f;
        if (s < S) dtv = ssd_softplus_clamp(scan_to_f(dt[(long)(b * S + s) * H + h]) + bias_h, dt_min, dt_max);
        sDt[t] = dtv;
    }
    __syncthreads();
    // cumsum（单线程，C 小）
    if (tid == 0) {
        float run = 0.f;
        #pragma unroll
        for (int t = 0; t < C; ++t) { run += A_h * sDt[t]; sAcs[t] = run; }
    }
    // 载 sB/sC
    for (int i = tid; i < C * N; i += nth) {
        const int t = i / N, n = i % N;
        const int s = base_s + t;
        if (s < S) {
            const long ib = (long)(b * S + s) * n_groups * N + (long)group * N + n;
            sB[i] = scan_to_f(Bin[ib]); sC[i] = scan_to_f(Cin[ib]);
        } else { sB[i] = 0.f; sC[i] = 0.f; }
    }
    // 载 sX = x·dt
    __syncthreads();
    for (int i = tid; i < C * P; i += nth) {
        const int t = i / P, p = i % P;
        const int s = base_s + t;
        sX[i] = (s < S) ? scan_to_f(x[(long)(b * S + s) * H * P + (long)h * P + p]) * sDt[t] : 0.f;
    }
    __syncthreads();

    // 写 Acs / Alast，预算 chunk_states 的衰减 sDecay[t]=exp(Acs[C-1]-Acs[t])（只依赖 t）
    {
        const float acs_last = sAcs[C - 1];
        for (int t = tid; t < C; t += nth) {
            Acs_g[(long)bhc * C + t] = sAcs[t];
            sDecay[t] = expf(acs_last - sAcs[t]);
        }
    }
    if (tid == 0) Alast_g[bhc] = sAcs[C - 1];

    // sM[t][s] = (s<=t) ? (Σ_n sC[t]·sB[s]) · exp(Acs[t]-Acs[s]) : 0
    for (int i = tid; i < C * C; i += nth) {
        const int t = i / C, s = i % C;
        if (s > t) { sM[i] = 0.f; continue; }
        float dot = 0.f;
        #pragma unroll
        for (int n = 0; n < N; ++n) dot += sC[t * N + n] * sB[s * N + n];
        sM[i] = dot * expf(sAcs[t] - sAcs[s]);
    }
    __syncthreads();

    // Y_diag[t][p] = Σ_{s<=t} sM[t][s]·sX[s][p]，+ D·x_raw，写 y
    for (int i = tid; i < C * P; i += nth) {
        const int t = i / P, p = i % P;
        const int s_out = base_s + t;
        if (s_out >= S) continue;
        float acc = 0.f;
        for (int s = 0; s <= t; ++s) acc += sM[t * C + s] * sX[s * P + p];
        const float x_raw = scan_to_f(x[(long)(b * S + s_out) * H * P + (long)h * P + p]);
        scan_store(&y[(long)(b * S + s_out) * H * P + (long)h * P + p], acc + Dh * x_raw);
    }

    // chunk_states[p][n] = Σ_t (sX[t][p]·sDecay[t])·sB[t][n]   （衰减预算，无 expf 内循环）
    __syncthreads();
    for (int i = tid; i < P * N; i += nth) {
        const int p = i / N, n = i % N;
        float acc = 0.f;
        #pragma unroll
        for (int t = 0; t < C; ++t)
            acc += sX[t * P + p] * sDecay[t] * sB[t * N + n];
        chunk_states[(long)bhc * P * N + i] = acc;
    }
}

// ---------------------------------------------------------------------------
// pass3：off-diagonal 修正。grid=(B*H*nc)，block=256。
//   Y_off[t][p] = (Σ_n sC[t][n]·S_in[p][n]) · exp(Acs[t])；y += Y_off
//   shared：sC[C*N] + sSin[P*N]（C=32 时 16KB+40KB=56KB → 需 opt-in，sm_89 支持）
// ---------------------------------------------------------------------------
template<typename IoT, int H, int P, int N, int C>
__global__ void ssd_fused_pass3(
    const float* __restrict__ Cin_f,  // 未用（C 从 IoT 读）
    const IoT* __restrict__ Cin,      // [B,S,G,N]
    const float* __restrict__ S_in,   // [B,H,nc,P,N]
    const float* __restrict__ Acs_g,  // [B,H,nc,C]
    IoT* __restrict__ y,              // [B,S,H,P]  in/out（+=）
    int batch, int S, int nc, int n_groups
) {
    extern __shared__ float smem[];
    float* sC   = smem;            // [C*N]
    float* sSin = smem + C * N;    // [P*N]

    const int bhc = blockIdx.x;
    const int c = bhc % nc;
    const int h = (bhc / nc) % H;
    const int b = bhc / (nc * H);
    const int group = h / (H / n_groups);
    const int base_s = c * C;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    for (int i = tid; i < C * N; i += nth) {
        const int t = i / N, n = i % N;
        const int s = base_s + t;
        sC[i] = (s < S) ? scan_to_f(Cin[(long)(b * S + s) * n_groups * N + (long)group * N + n]) : 0.f;
    }
    for (int i = tid; i < P * N; i += nth) sSin[i] = S_in[(long)bhc * P * N + i];
    __syncthreads();

    for (int i = tid; i < C * P; i += nth) {
        const int t = i / P, p = i % P;
        const int s_out = base_s + t;
        if (s_out >= S) continue;
        float dot = 0.f;
        #pragma unroll
        for (int n = 0; n < N; ++n) dot += sC[t * N + n] * sSin[p * N + n];
        const float yoff = dot * expf(Acs_g[(long)bhc * C + t]);
        const long yi = (long)(b * S + s_out) * H * P + (long)h * P + p;
        scan_store(&y[yi], scan_to_f(y[yi]) + yoff);
    }
}

// ===========================================================================
// host wrapper
// ===========================================================================
template<typename IoT, int H, int P, int N, int C>
inline void ssd_scan_fused_prefill(
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
    auto cstate = wf(BHC * P * N);
    auto Acs    = wf(BHC * C);
    auto Alast  = wf(BHC);
    auto PA     = wf(BH * (nc + 1));
    auto Tm     = wf(BH * nc * nc);
    auto Tlast  = wf(BH * nc);
    auto S_in   = wf(BHC * P * N);

    const int TPB = 256;
    auto ngrid = [&](long n) { return (int)((n + TPB - 1) / TPB); };

    // pass1
    ssd_fused_pass1<IoT, H, P, N, C><<<(int)BHC, TPB, 0, stream>>>(
        x, dt, B, Cmat, A_log, D_param, dt_bias, y,
        cstate.data_, Acs.data_, Alast.data_, batch, S, nc, n_groups, dt_min, dt_max);

    // inter-chunk（复用 chunked 的 T 矩阵 + gemm）
    ssdc_prefix_A<<<ngrid(BH), TPB, 0, stream>>>(Alast.data_, PA.data_, (int)BH, nc);
    ssdc_build_T<<<ngrid(BH * (long)nc * nc), TPB, 0, stream>>>(
        PA.data_, Tm.data_, Tlast.data_, BH, nc);
    gemm_strided_batched_tf32(Tm.data_, cstate.data_, S_in.data_,
        nc, P * N, nc, false, false,
        (long long)nc * nc, (long long)nc * P * N, (long long)nc * P * N, (int)BH, stream);
    if (ssm_state_out) {
        gemm_strided_batched_tf32(Tlast.data_, cstate.data_, ssm_state_out,
            1, P * N, nc, false, false,
            (long long)nc, (long long)nc * P * N, (long long)P * N, (int)BH, stream);
    }

    // pass3（opt-in 动态 shared）
    const size_t shmem = (size_t)(C * N + P * N) * sizeof(float);
    static bool s_opt = false;
    if (!s_opt) {
        cudaFuncSetAttribute(ssd_fused_pass3<IoT, H, P, N, C>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        s_opt = true;
    }
    ssd_fused_pass3<IoT, H, P, N, C><<<(int)BHC, TPB, shmem, stream>>>(
        nullptr, Cmat, S_in.data_, Acs.data_, y, batch, S, nc, n_groups);
}

template<int H, int P, int N, int C>
inline void ssd_scan_fused_prefill_fp32(
    const float* x, const float* dt, const float* A_log,
    const float* B, const float* C_, const float* D_param, const float* dt_bias,
    float* y, float* ssm_state_out, int batch, int S, int n_groups,
    float dt_min = 0.f, float dt_max = FLT_MAX, cudaStream_t stream = nullptr
) {
    ssd_scan_fused_prefill<float, H, P, N, C>(x, dt, A_log, B, C_, D_param, dt_bias,
        y, ssm_state_out, batch, S, n_groups, dt_min, dt_max, stream);
}

template<int H, int P, int N, int C>
inline void ssd_scan_fused_prefill_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* dt, const float* A_log,
    const __nv_bfloat16* B, const __nv_bfloat16* C_, const float* D_param, const float* dt_bias,
    __nv_bfloat16* y, float* ssm_state_out, int batch, int S, int n_groups,
    float dt_min = 0.f, float dt_max = FLT_MAX, cudaStream_t stream = nullptr
) {
    ssd_scan_fused_prefill<__nv_bfloat16, H, P, N, C>(x, dt, A_log, B, C_, D_param, dt_bias,
        y, ssm_state_out, batch, S, n_groups, dt_min, dt_max, stream);
}

}  // namespace nemotron::ops::mamba2

#endif  // NEMOTRON_INFER_MAMBA2_SSD_SCAN_FUSED_CUH

//
// Created by Administrator on 2026/6/15.
//
// SDPA Decode — 单 query token 对整段 KV cache 做注意力。
//
// 采用 FlashDecoding split-K：
//   Phase 1 (split)：把 KV 序列 S 切成 nsplit 段，1 warp = 1 个 (head, split)，
//     每 lane 持 HEAD/32 个 head_dim，点积用单次 warp 蝶式 all-reduce
//     （无 __syncthreads、无 shared mem），每段算出局部 online-softmax 的
//     偏量 (m, l, o[HEAD])。H×nsplit 个 warp 铺满 GPU。
//   Phase 2 (combine)：每个 head 一个 warp，把 nsplit 段偏量按标准 online-softmax
//     合并归一，并把 K_new/V_new 追加到 cache 末尾。
//
// 旧实现是 <<<H,128>>> 单 block/head + 每位置 2 次 block barrier，延迟瓶颈且
// GPU 严重空转（H=4 只用 4 个 SM），BF16 无加速。split-K 同时解决两者。
//
#ifndef NEMOTRON_INFER_SDPA_DECODE_CUH
#define NEMOTRON_INFER_SDPA_DECODE_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#ifdef USE_CUDNN
#include "sdpa_cudnn.h"
#endif

namespace nemotron::ops::attention {

// ===========================================================================
// 类型转换 helper（fp32 / bf16 统一 kernel）
// ===========================================================================
template<typename T> __device__ __forceinline__ float dec_to_float(T x);
template<> __device__ __forceinline__ float dec_to_float<float>(float x) { return x; }
template<> __device__ __forceinline__ float dec_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<typename T> __device__ __forceinline__ T dec_from_float(float x);
template<> __device__ __forceinline__ float dec_from_float<float>(float x) { return x; }
template<> __device__ __forceinline__ __nv_bfloat16 dec_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;  // 蝶式：所有 lane 得到相同结果
}

// ===========================================================================
// Phase 1: split — 每 warp 处理 (head, split) 的一段 KV，输出局部 (m,l,o)
//   m_part [H*nsplit], l_part [H*nsplit], o_part [H*nsplit*HEAD] (均 fp32)
// ===========================================================================
template<typename T, int HEAD = 128>
__global__ void sdpa_decode_split_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K_cache,
    const T* __restrict__ V_cache,
    const int S_cache,
    const int total_S,
    const int num_heads,
    const int nsplit,
    const float rsqrt_d,
    float* __restrict__ m_part,
    float* __restrict__ l_part,
    float* __restrict__ o_part
) {
    constexpr int DPL = HEAD / 32;   // 每 lane 持有的 head_dim 数（128/32=4）
    const int warp_in_block = threadIdx.x >> 5;
    const int gwid = blockIdx.x * (blockDim.x >> 5) + warp_in_block;  // 全局 warp id
    if (gwid >= num_heads * nsplit) return;

    const int head  = gwid / nsplit;
    const int split = gwid % nsplit;
    const int lane  = threadIdx.x & 31;

    const int chunk   = (S_cache + nsplit - 1) / nsplit;
    const int s_begin = split * chunk;
    int s_end = s_begin + chunk;
    if (s_end > S_cache) s_end = S_cache;

    // Q：每 lane 取 DPL 个连续 dim（lane*DPL .. +DPL-1），跨 lane 即 0..HEAD-1
    const T* Qh = Q + (size_t)head * HEAD;
    float q_reg[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) q_reg[i] = dec_to_float<T>(Qh[lane * DPL + i]);

    const size_t kv_base = (size_t)head * total_S * HEAD;
    float m = -FLT_MAX, l = 0.f;
    float o[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) o[i] = 0.f;

    for (int s = s_begin; s < s_end; ++s) {
        const T* Ks = K_cache + kv_base + (size_t)s * HEAD;
        float dot = 0.f;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) dot += q_reg[i] * dec_to_float<T>(Ks[lane * DPL + i]);
        dot = warp_allreduce_sum(dot);          // 完整点积，所有 lane 一致
        const float score = dot * rsqrt_d;

        const float m_new = fmaxf(m, score);
        const float corr  = expf(m - m_new);
        const float p     = expf(score - m_new);
        const T* Vs = V_cache + kv_base + (size_t)s * HEAD;
        #pragma unroll
        for (int i = 0; i < DPL; ++i)
            o[i] = o[i] * corr + p * dec_to_float<T>(Vs[lane * DPL + i]);
        l = l * corr + p;
        m = m_new;
    }

    const int pidx = gwid;  // head*nsplit + split
    if (lane == 0) { m_part[pidx] = m; l_part[pidx] = l; }
    float* op = o_part + (size_t)pidx * HEAD;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) op[lane * DPL + i] = o[i];
}

// ===========================================================================
// Phase 2: combine — 每 head 一个 warp，合并 nsplit 段偏量 + 追加 K_new/V_new
// ===========================================================================
template<typename T, int HEAD = 128>
__global__ void sdpa_decode_combine_kernel(
    T* __restrict__ O,
    T* __restrict__ K_cache,
    T* __restrict__ V_cache,
    const T* __restrict__ K_new,
    const T* __restrict__ V_new,
    const float* __restrict__ m_part,
    const float* __restrict__ l_part,
    const float* __restrict__ o_part,
    const int num_heads,
    const int nsplit,
    const int S_cache,
    const int total_S
) {
    constexpr int DPL = HEAD / 32;
    const int head = blockIdx.x;
    if (head >= num_heads) return;
    const int lane = threadIdx.x & 31;

    // 全局 max
    float gm = -FLT_MAX;
    for (int sp = 0; sp < nsplit; ++sp) gm = fmaxf(gm, m_part[head * nsplit + sp]);

    // 合并各段：l = Σ exp(m_i-gm)·l_i, o = Σ exp(m_i-gm)·o_i
    float gl = 0.f;
    float acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.f;
    for (int sp = 0; sp < nsplit; ++sp) {
        const int pidx = head * nsplit + sp;
        const float w = expf(m_part[pidx] - gm);
        gl += w * l_part[pidx];
        const float* op = o_part + (size_t)pidx * HEAD;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] += w * op[lane * DPL + i];
    }
    const float inv = (gl > 0.f) ? 1.f / gl : 0.f;

    T* Oh = O + (size_t)head * HEAD;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) Oh[lane * DPL + i] = dec_from_float<T>(acc[i] * inv);

    // 追加新 KV 到 cache 末尾（位置 S_cache）
    T* Kc = K_cache + (size_t)head * total_S * HEAD + (size_t)S_cache * HEAD;
    T* Vc = V_cache + (size_t)head * total_S * HEAD + (size_t)S_cache * HEAD;
    const T* Kn = K_new + (size_t)head * HEAD;
    const T* Vn = V_new + (size_t)head * HEAD;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) {
        Kc[lane * DPL + i] = Kn[lane * DPL + i];
        Vc[lane * DPL + i] = Vn[lane * DPL + i];
    }
}

// ===========================================================================
// 内部持久 scratch（按需增长）。decode 是逐 token 热路径，每次 cudaMalloc 不可取；
// scratch 是 split-K 的实现细节、尺寸小（H*nsplit*HEAD float），故进程级持有。
// ===========================================================================
struct DecodeScratch {
    float* m = nullptr;
    float* l = nullptr;
    float* o = nullptr;
    size_t ml_cap = 0;
    size_t o_cap  = 0;
};

inline DecodeScratch& decode_scratch() {
    static DecodeScratch s;
    return s;
}

inline void decode_ensure_scratch(int H, int nsplit, int HEAD) {
    auto& s = decode_scratch();
    const size_t ml = (size_t)H * nsplit;
    const size_t oc = ml * HEAD;
    if (ml > s.ml_cap) {
        if (s.m) cudaFree(s.m);
        if (s.l) cudaFree(s.l);
        cudaMalloc(&s.m, ml * sizeof(float));
        cudaMalloc(&s.l, ml * sizeof(float));
        s.ml_cap = ml;
    }
    if (oc > s.o_cap) {
        if (s.o) cudaFree(s.o);
        cudaMalloc(&s.o, oc * sizeof(float));
        s.o_cap = oc;
    }
}

// ===========================================================================
// dispatch：选 nsplit、launch phase1 + phase2
// ===========================================================================
template<typename T, int HEAD>
inline void sdpa_decode_dispatch(
    const T* Q, T* K_cache, T* V_cache, T* O,
    const T* K_new, const T* V_new,
    int S_cache, int total_S, int num_heads, cudaStream_t stream
) {
    if (num_heads == 0) return;
    const float rsqrt_d = 1.f / sqrtf((float)HEAD);

    // 每段约 128 个 KV 位置，nsplit 上限 64（铺满 GPU 又不过度切分）
    int nsplit = (S_cache + 127) / 128;
    if (nsplit < 1) nsplit = 1;
    if (nsplit > 64) nsplit = 64;

    decode_ensure_scratch(num_heads, nsplit, HEAD);
    auto& s = decode_scratch();

    const int warps_per_block = 4;
    const int total_warps = num_heads * nsplit;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    sdpa_decode_split_kernel<T, HEAD><<<blocks, warps_per_block * 32, 0, stream>>>(
        Q, K_cache, V_cache, S_cache, total_S, num_heads, nsplit, rsqrt_d,
        s.m, s.l, s.o
    );
    sdpa_decode_combine_kernel<T, HEAD><<<num_heads, 32, 0, stream>>>(
        O, K_cache, V_cache, K_new, V_new, s.m, s.l, s.o,
        num_heads, nsplit, S_cache, total_S
    );
}

// ===========================================================================
// __host__ 入口 — 签名与测试保持一致
// ===========================================================================
template<int HEAD = 128>
inline __host__ void sdpa_decode_fp32(
    const float* Q, float* K_cache, float* V_cache, float* O,
    const float* K_new, const float* V_new,
    const int S_cache, const int total_S, const int num_heads,
    cudaStream_t stream = nullptr
) {
    sdpa_decode_dispatch<float, HEAD>(
        Q, K_cache, V_cache, O, K_new, V_new, S_cache, total_S, num_heads, stream);
}

template<int HEAD = 128>
inline __host__ void sdpa_decode_bf16(
    const __nv_bfloat16* Q, __nv_bfloat16* K_cache, __nv_bfloat16* V_cache,
    __nv_bfloat16* O, const __nv_bfloat16* K_new, const __nv_bfloat16* V_new,
    const int S_cache, const int total_S, const int num_heads,
    cudaStream_t stream = nullptr
) {
    // cuDNN 的 SDPA decode 需把 K_new/V_new 拼入 cache 再走 graph，且单 token
    // launch 开销大；此处用手写 split-K（已是 bandwidth-bound）。
    sdpa_decode_dispatch<__nv_bfloat16, HEAD>(
        Q, K_cache, V_cache, O, K_new, V_new, S_cache, total_S, num_heads, stream);
}

}  // namespace nemotron::ops::attention

#endif //NEMOTRON_INFER_SDPA_DECODE_CUH

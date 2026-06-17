//
// attention_block.cuh — Nemotron-H Attention 类型 DecoderBlock（prefill + decode）
// ===========================================================================
// 组装单层 `NemotronHBlock`(block_type=attention)：
//   out = input + mixer( rmsnorm(input) )            ← pre-norm 残差
// mixer = NemotronHAttention.forward:
//   q/k/v_proj → (reshape head) → causal SDPA(GQA) → o_proj
//
// ⚠️ Nemotron-H attention 是 NoPE（无 RoPE / 无位置编码）：位置信息完全由
//    Mamba 层承担。故本 block 不含任何 rotary，纯 QKV→SDPA→O。
//
// 形状（真实模型）：HIDDEN=3136, H_Q=40, H_KV=8（GQA group=5）, HEAD=128。
//   q_proj [H_Q*HEAD, HIDDEN]=[5120,3136]，k/v_proj [H_KV*HEAD,HIDDEN]=[1024,3136]，
//   o_proj [HIDDEN, H_Q*HEAD]=[3136,5120]。scale=1/sqrt(HEAD)。无 bias。
//
// 布局：proj 输出 [S, H*HEAD] 行主序 = [S,H,HEAD]；SDPA 算子要 head-major
//   [H,S,HEAD]，故 prefill 需 transpose（M=S>1）。decode M=1 时 [1,H,HEAD]≡[H,HEAD]
//   天然连续，免 transpose。
//
// KV cache（decode 续接）：布局 [H_KV, total_S, HEAD]。prefill 把 K/V 写入
//   [0,S)；decode 每步 sdpa_decode 内部追加新 token 到位置 S_cache。
// ===========================================================================
#ifndef NEMOTRON_INFER_MODEL_ATTENTION_BLOCK_CUH
#define NEMOTRON_INFER_MODEL_ATTENTION_BLOCK_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tensor/tensor.h"
#include "ops/reduce.cuh"
#include "ops/gemm.cuh"
#include "ops/gemv.cuh"
#include "ops/elementwise.cuh"
#include "ops/attention/sdpa_prefill.cuh"
#include "ops/attention/sdpa_decode.cuh"

namespace nemotron::model {

// ===========================================================================
// transpose 核：[S, H, D] (row-major, seq-major) ↔ [H, S, D] (head-major)
//   每线程搬 1 个元素；grid-stride。HEAD=D 连续。
// ===========================================================================
template<typename T>
__global__ void attn_transpose_shd_to_hsd(
    const T* __restrict__ in,   // [S, H, D]
    T* __restrict__ out,        // [H, S, D]
    int S, int Hh, int D
) {
    const size_t total = (size_t)S * Hh * D;
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         i < total; i += (size_t)blockDim.x * gridDim.x) {
        const int d = i % D;
        const int h = (i / D) % Hh;
        const int s = i / ((size_t)D * Hh);
        out[((size_t)h * S + s) * D + d] = in[i];
    }
}

template<typename T>
__global__ void attn_transpose_hsd_to_shd(
    const T* __restrict__ in,   // [H, S, D]
    T* __restrict__ out,        // [S, H, D]
    int S, int Hh, int D
) {
    const size_t total = (size_t)S * Hh * D;
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         i < total; i += (size_t)blockDim.x * gridDim.x) {
        const int d = i % D;
        const int s = (i / D) % S;
        const int h = i / ((size_t)D * S);
        out[((size_t)s * Hh + h) * D + d] = in[i];
    }
}

template<typename T>
inline void attn_transpose_shd_to_hsd_launch(const T* in, T* out, int S, int Hh, int D, cudaStream_t stream) {
    const size_t total = (size_t)S * Hh * D;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    attn_transpose_shd_to_hsd<T><<<blocks, threads, 0, stream>>>(in, out, S, Hh, D);
}

template<typename T>
inline void attn_transpose_hsd_to_shd_launch(const T* in, T* out, int S, int Hh, int D, cudaStream_t stream) {
    const size_t total = (size_t)S * Hh * D;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    attn_transpose_hsd_to_shd<T><<<blocks, threads, 0, stream>>>(in, out, S, Hh, D);
}

// ===========================================================================
// FP32 参考路径（prefill）
// ===========================================================================
struct AttnBlockWeightsFP32 {
    const float* block_norm_w;   // [HIDDEN]
    const float* q_proj_w;       // [H_Q*HEAD, HIDDEN]
    const float* k_proj_w;       // [H_KV*HEAD, HIDDEN]
    const float* v_proj_w;       // [H_KV*HEAD, HIDDEN]
    const float* o_proj_w;       // [HIDDEN, H_Q*HEAD]
};

template<int HIDDEN, int H_Q, int H_KV, int HEAD = 128>
inline void attention_block_forward_fp32(
    const float* input,          // [S, HIDDEN]   (B=1)
    const AttnBlockWeightsFP32& w,
    float* out,                  // [S, HIDDEN]
    int S,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::attention;
    constexpr int QD = H_Q * HEAD;
    constexpr int KD = H_KV * HEAD;
    constexpr float EPS = 1e-5f;
    const int M = S;

    auto normed = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));
    auto q      = allocate_tensor<float>(TensorShape::make_2d(M, QD));
    auto k      = allocate_tensor<float>(TensorShape::make_2d(M, KD));
    auto v      = allocate_tensor<float>(TensorShape::make_2d(M, KD));
    auto qt     = allocate_tensor<float>(TensorShape::make_2d(M, QD));
    auto kt     = allocate_tensor<float>(TensorShape::make_2d(M, KD));
    auto vt     = allocate_tensor<float>(TensorShape::make_2d(M, KD));
    auto ot     = allocate_tensor<float>(TensorShape::make_2d(M, QD));
    auto o      = allocate_tensor<float>(TensorShape::make_2d(M, QD));
    auto attn   = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));

    rmsnorm_fp32(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);
    gemm_fp32(normed.data_, w.q_proj_w, q.data_, M, QD, HIDDEN, true, stream);
    gemm_fp32(normed.data_, w.k_proj_w, k.data_, M, KD, HIDDEN, true, stream);
    gemm_fp32(normed.data_, w.v_proj_w, v.data_, M, KD, HIDDEN, true, stream);

    attn_transpose_shd_to_hsd_launch<float>(q.data_, qt.data_, S, H_Q,  HEAD, stream);
    attn_transpose_shd_to_hsd_launch<float>(k.data_, kt.data_, S, H_KV, HEAD, stream);
    attn_transpose_shd_to_hsd_launch<float>(v.data_, vt.data_, S, H_KV, HEAD, stream);

    sdpa_prefill_fp32<32, HEAD>(qt.data_, kt.data_, vt.data_, ot.data_, S, H_Q, H_KV, stream);

    attn_transpose_hsd_to_shd_launch<float>(ot.data_, o.data_, S, H_Q, HEAD, stream);
    gemm_fp32(o.data_, w.o_proj_w, attn.data_, M, HIDDEN, QD, true, stream);

    elementwise_ops_fp32<kElementwiseAdd>(
        const_cast<float*>(input), attn.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// BF16 路径
// ===========================================================================
struct AttnBlockWeightsBF16 {
    const float*         block_norm_w;   // [HIDDEN]            fp32
    const __nv_bfloat16* q_proj_w;       // [H_Q*HEAD, HIDDEN]  bf16
    const __nv_bfloat16* k_proj_w;       // [H_KV*HEAD, HIDDEN] bf16
    const __nv_bfloat16* v_proj_w;       // [H_KV*HEAD, HIDDEN] bf16
    const __nv_bfloat16* o_proj_w;       // [HIDDEN, H_Q*HEAD]  bf16
};

// KV cache：[H_KV, cache_cap, HEAD] bf16。prefill 把 K/V 写到 [0,S)。
//   cache_cap 是分配容量（>=S；decode 续接时 = 最大序列长度）。
template<int HIDDEN, int H_Q, int H_KV, int HEAD = 128>
inline void attention_block_forward_bf16(
    const __nv_bfloat16* input,          // [S, HIDDEN]
    const AttnBlockWeightsBF16& w,
    __nv_bfloat16* out,                  // [S, HIDDEN]
    int S,
    __nv_bfloat16* k_cache = nullptr,    // [H_KV, cache_cap, HEAD] 写 [0,S)
    __nv_bfloat16* v_cache = nullptr,
    int cache_cap = 0,                   // cache 第二维容量（0 → =S）
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::attention;
    using bf16 = __nv_bfloat16;
    constexpr int QD = H_Q * HEAD;
    constexpr int KD = H_KV * HEAD;
    constexpr float EPS = 1e-5f;
    const int M = S;
    if (cache_cap <= 0) cache_cap = S;

    auto normed = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto q      = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto k      = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto v      = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto qt     = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto kt     = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto vt     = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto ot     = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto o      = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto attn   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));

    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);
    gemm_bf16(normed.data_, w.q_proj_w, q.data_, M, QD, HIDDEN, stream);
    gemm_bf16(normed.data_, w.k_proj_w, k.data_, M, KD, HIDDEN, stream);
    gemm_bf16(normed.data_, w.v_proj_w, v.data_, M, KD, HIDDEN, stream);

    attn_transpose_shd_to_hsd_launch<bf16>(q.data_, qt.data_, S, H_Q,  HEAD, stream);
    attn_transpose_shd_to_hsd_launch<bf16>(k.data_, kt.data_, S, H_KV, HEAD, stream);
    attn_transpose_shd_to_hsd_launch<bf16>(v.data_, vt.data_, S, H_KV, HEAD, stream);

    sdpa_prefill_bf16<32, HEAD>(qt.data_, kt.data_, vt.data_, ot.data_, S, H_Q, H_KV, stream);

    attn_transpose_hsd_to_shd_launch<bf16>(ot.data_, o.data_, S, H_Q, HEAD, stream);
    gemm_bf16(o.data_, w.o_proj_w, attn.data_, M, HIDDEN, QD, stream);

    elementwise_ops_bf16<kElementwiseAdd>(
        input, attn.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);

    // KV cache 写入 [0,S)：kt/vt 是 [H_KV,S,HEAD]，cache 是 [H_KV,cache_cap,HEAD]
    if (k_cache && v_cache) {
        cudaMemcpy2DAsync(k_cache, (size_t)cache_cap * HEAD * sizeof(bf16),
                          kt.data_, (size_t)S * HEAD * sizeof(bf16),
                          (size_t)S * HEAD * sizeof(bf16), H_KV,
                          cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(v_cache, (size_t)cache_cap * HEAD * sizeof(bf16),
                          vt.data_, (size_t)S * HEAD * sizeof(bf16),
                          (size_t)S * HEAD * sizeof(bf16), H_KV,
                          cudaMemcpyDeviceToDevice, stream);
    }
}

// ===========================================================================
// BF16 decode：单 token，续接 KV cache。
//   q/k/v_proj 输出 [1,H,HEAD]≡[H,HEAD] 连续，免 transpose。
//   sdpa_decode 内部把 k_new/v_new 追加到 cache 位置 S_cache。
// ===========================================================================
template<int HIDDEN, int H_Q, int H_KV, int HEAD = 128>
inline void attention_block_decode_bf16(
    const __nv_bfloat16* input,          // [1, HIDDEN]
    const AttnBlockWeightsBF16& w,
    __nv_bfloat16* k_cache,              // [H_KV, cache_cap, HEAD] in/out
    __nv_bfloat16* v_cache,              // [H_KV, cache_cap, HEAD] in/out
    int S_cache,                         // 已缓存长度（新 token 落在位置 S_cache）
    int cache_cap,                       // cache 第二维容量（= total_S）
    __nv_bfloat16* out,                  // [1, HIDDEN]
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::attention;
    using bf16 = __nv_bfloat16;
    constexpr int QD = H_Q * HEAD;
    constexpr int KD = H_KV * HEAD;
    constexpr float EPS = 1e-5f;

    auto normed = allocate_tensor<bf16>(TensorShape::make_2d(1, HIDDEN));
    auto q      = allocate_tensor<bf16>(TensorShape::make_2d(1, QD));
    auto knew   = allocate_tensor<bf16>(TensorShape::make_2d(1, KD));
    auto vnew   = allocate_tensor<bf16>(TensorShape::make_2d(1, KD));
    auto ot     = allocate_tensor<bf16>(TensorShape::make_2d(1, QD));
    auto attn   = allocate_tensor<bf16>(TensorShape::make_2d(1, HIDDEN));

    rmsnorm_bf16(input, normed.data_, w.block_norm_w, 1, HIDDEN, EPS, stream);
    gemm_bf16(normed.data_, w.q_proj_w, q.data_,    1, QD, HIDDEN, stream);
    gemm_bf16(normed.data_, w.k_proj_w, knew.data_, 1, KD, HIDDEN, stream);
    gemm_bf16(normed.data_, w.v_proj_w, vnew.data_, 1, KD, HIDDEN, stream);

    // cache 的 total_S 维 = cache_cap；split kernel 用 total_S 计算 cache 行跨距
    sdpa_decode_bf16<HEAD>(
        q.data_, k_cache, v_cache, ot.data_, knew.data_, vnew.data_,
        S_cache, cache_cap, H_Q, H_KV, stream);

    gemm_bf16(ot.data_, w.o_proj_w, attn.data_, 1, HIDDEN, QD, stream);
    elementwise_ops_bf16<kElementwiseAdd>(
        input, attn.data_, out, (size_t)HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// FP8 路径（q/k/v/o_proj 走原生 e4m3；SDPA 仍 bf16 内部 fp32）
//   normed 量化一次复用给 q/k/v 三个 proj（同一激活，不同权重）。
// ===========================================================================
struct AttnBlockWeightsFP8 {
    const float*         block_norm_w;     // [HIDDEN]            fp32
    const __nv_fp8_e4m3* q_proj_w;         // [H_Q*HEAD, HIDDEN]  e4m3
    const float*         q_proj_wscale;    // [H_Q*HEAD]
    const __nv_fp8_e4m3* k_proj_w;         // [H_KV*HEAD, HIDDEN] e4m3
    const float*         k_proj_wscale;    // [H_KV*HEAD]
    const __nv_fp8_e4m3* v_proj_w;         // [H_KV*HEAD, HIDDEN] e4m3
    const float*         v_proj_wscale;    // [H_KV*HEAD]
    const __nv_fp8_e4m3* o_proj_w;         // [HIDDEN, H_Q*HEAD]  e4m3
    const float*         o_proj_wscale;    // [HIDDEN]
};

template<int HIDDEN, int H_Q, int H_KV, int HEAD = 128>
inline void attention_block_forward_fp8(
    const __nv_bfloat16* input,          // [S, HIDDEN]
    const AttnBlockWeightsFP8& w,
    __nv_bfloat16* out,                  // [S, HIDDEN]
    int S,
    __nv_bfloat16* k_cache = nullptr,
    __nv_bfloat16* v_cache = nullptr,
    int cache_cap = 0,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::attention;
    using bf16 = __nv_bfloat16;
    using fp8  = __nv_fp8_e4m3;
    constexpr int QD = H_Q * HEAD;
    constexpr int KD = H_KV * HEAD;
    constexpr float EPS = 1e-5f;
    constexpr size_t WS_BYTES = 32ull * 1024 * 1024;
    const int M = S;
    if (cache_cap <= 0) cache_cap = S;

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto norm_fp8 = allocate_tensor<fp8>(TensorShape::make_2d(M, HIDDEN));
    auto q        = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto k        = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto v        = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto qt       = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto kt       = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto vt       = allocate_tensor<bf16>(TensorShape::make_2d(M, KD));
    auto ot       = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto o        = allocate_tensor<bf16>(TensorShape::make_2d(M, QD));
    auto o_fp8    = allocate_tensor<fp8>(TensorShape::make_2d(M, QD));
    auto attn     = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto xscale1  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto xscale2  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto amax     = allocate_tensor<unsigned int>(TensorShape::make_1d(1));
    auto ws       = allocate_tensor<char>(TensorShape::make_1d((int64_t)WS_BYTES));

    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    quantize_activation_fp8(normed.data_, norm_fp8.data_, xscale1.data_,
                            amax.data_, (size_t)M * HIDDEN, stream);
    gemm_fp8(norm_fp8.data_, w.q_proj_w, q.data_, xscale1.data_, w.q_proj_wscale,
             M, QD, HIDDEN, ws.data_, WS_BYTES, stream);
    gemm_fp8(norm_fp8.data_, w.k_proj_w, k.data_, xscale1.data_, w.k_proj_wscale,
             M, KD, HIDDEN, ws.data_, WS_BYTES, stream);
    gemm_fp8(norm_fp8.data_, w.v_proj_w, v.data_, xscale1.data_, w.v_proj_wscale,
             M, KD, HIDDEN, ws.data_, WS_BYTES, stream);

    attn_transpose_shd_to_hsd_launch<bf16>(q.data_, qt.data_, S, H_Q,  HEAD, stream);
    attn_transpose_shd_to_hsd_launch<bf16>(k.data_, kt.data_, S, H_KV, HEAD, stream);
    attn_transpose_shd_to_hsd_launch<bf16>(v.data_, vt.data_, S, H_KV, HEAD, stream);

    sdpa_prefill_bf16<32, HEAD>(qt.data_, kt.data_, vt.data_, ot.data_, S, H_Q, H_KV, stream);

    attn_transpose_hsd_to_shd_launch<bf16>(ot.data_, o.data_, S, H_Q, HEAD, stream);

    quantize_activation_fp8(o.data_, o_fp8.data_, xscale2.data_,
                            amax.data_, (size_t)M * QD, stream);
    gemm_fp8(o_fp8.data_, w.o_proj_w, attn.data_, xscale2.data_, w.o_proj_wscale,
             M, HIDDEN, QD, ws.data_, WS_BYTES, stream);

    elementwise_ops_bf16<kElementwiseAdd>(
        input, attn.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);

    if (k_cache && v_cache) {
        cudaMemcpy2DAsync(k_cache, (size_t)cache_cap * HEAD * sizeof(bf16),
                          kt.data_, (size_t)S * HEAD * sizeof(bf16),
                          (size_t)S * HEAD * sizeof(bf16), H_KV,
                          cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(v_cache, (size_t)cache_cap * HEAD * sizeof(bf16),
                          vt.data_, (size_t)S * HEAD * sizeof(bf16),
                          (size_t)S * HEAD * sizeof(bf16), H_KV,
                          cudaMemcpyDeviceToDevice, stream);
    }
}

// ===========================================================================
// FP8 decode：q/k/v/o_proj 走 e4m3，SDPA decode 仍 bf16。
// ===========================================================================
template<int HIDDEN, int H_Q, int H_KV, int HEAD = 128>
inline void attention_block_decode_fp8(
    const __nv_bfloat16* input,          // [1, HIDDEN]
    const AttnBlockWeightsFP8& w,
    __nv_bfloat16* k_cache,              // [H_KV, cache_cap, HEAD] in/out
    __nv_bfloat16* v_cache,
    int S_cache,
    int cache_cap,
    __nv_bfloat16* out,                  // [1, HIDDEN]
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::attention;
    using bf16 = __nv_bfloat16;
    using fp8  = __nv_fp8_e4m3;
    constexpr int QD = H_Q * HEAD;
    constexpr int KD = H_KV * HEAD;
    constexpr float EPS = 1e-5f;

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(1, HIDDEN));
    auto q        = allocate_tensor<bf16>(TensorShape::make_2d(1, QD));
    auto knew     = allocate_tensor<bf16>(TensorShape::make_2d(1, KD));
    auto vnew     = allocate_tensor<bf16>(TensorShape::make_2d(1, KD));
    auto ot       = allocate_tensor<bf16>(TensorShape::make_2d(1, QD));
    auto attn     = allocate_tensor<bf16>(TensorShape::make_2d(1, HIDDEN));

    rmsnorm_bf16(input, normed.data_, w.block_norm_w, 1, HIDDEN, EPS, stream);

    // q/k/v_proj: M=1 fp8 gemv（激活 bf16，W8A16）
    gemv_fp8(normed.data_, w.q_proj_w, w.q_proj_wscale, q.data_,    QD, HIDDEN, stream);
    gemv_fp8(normed.data_, w.k_proj_w, w.k_proj_wscale, knew.data_, KD, HIDDEN, stream);
    gemv_fp8(normed.data_, w.v_proj_w, w.v_proj_wscale, vnew.data_, KD, HIDDEN, stream);

    sdpa_decode_bf16<HEAD>(
        q.data_, k_cache, v_cache, ot.data_, knew.data_, vnew.data_,
        S_cache, cache_cap, H_Q, H_KV, stream);

    // o_proj: M=1 fp8 gemv
    gemv_fp8(ot.data_, w.o_proj_w, w.o_proj_wscale, attn.data_, HIDDEN, QD, stream);

    elementwise_ops_bf16<kElementwiseAdd>(
        input, attn.data_, out, (size_t)HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

}  // namespace nemotron::model

#endif  // NEMOTRON_INFER_MODEL_ATTENTION_BLOCK_CUH

//
// binding.cu — Nemotron-H 整模型 Python 编排绑定（bf16 路径）
// ===========================================================================
// 设计：Python 持久状态（hidden ping-pong / KV / conv / ssm cache）+ 权重，
//   逐层调用本模块导出的 CUDA block。所有 device 指针以 uintptr_t 传入
//   （Python 侧用 torch tensor.data_ptr()）。
//
//   ⚠️ 每层 forward/decode 调用前 Python 须先 reset_allocator()：block 的瞬态
//      workspace 走全局 BumpAllocator，输出写到调用方持久 buffer。
//
//   维度走真实模型 config（编译期常量，模板实例化）。
// ===========================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/embedding.cuh"
#include "ops/reduce.cuh"
#include "ops/gemm.cuh"
#include "model/mamba_block.cuh"
#include "model/attention_block.cuh"
#include "model/mlp_block.cuh"

namespace py = pybind11;
using namespace nemotron;
using namespace nemotron::model;
using bf16 = __nv_bfloat16;

// ---- 真实模型 config ----
namespace cfg {
constexpr int HIDDEN = 3136;
// Mamba2
constexpr int H = 96, P = 80, N = 128, G = 8, CONV_K = 4;
// Attention
constexpr int H_Q = 40, H_KV = 8, HEAD = 128;
// MLP
constexpr int INTER = 12544;
// 词表
constexpr int VOCAB = 131072;
constexpr float EPS = 1e-5f;
}

using fp8 = __nv_fp8_e4m3;
static inline bf16*        b(uintptr_t p) { return reinterpret_cast<bf16*>(p); }
static inline const bf16*  cb(uintptr_t p){ return reinterpret_cast<const bf16*>(p); }
static inline float*       f(uintptr_t p) { return reinterpret_cast<float*>(p); }
static inline const float* cf(uintptr_t p){ return reinterpret_cast<const float*>(p); }
static inline const fp8*   c8(uintptr_t p){ return reinterpret_cast<const fp8*>(p); }

// ===========================================================================
// 基础 op
// ===========================================================================
// 软回退：保留 slab 只挪偏移，避免每层 cudaFree/cudaMalloc 同步阻塞（decode 真瓶颈）。
static void reset_allocator() { default_allocator().rewind(); }
// 真正释放全部 slab（如需在大 prefill 后回收显存可显式调用）。
static void free_allocator() { default_allocator().reset(); }
static void sync() { cudaDeviceSynchronize(); }

// embedding: ids[int64, num_tokens] → out[bf16, num_tokens, HIDDEN]
static void embedding(uintptr_t table, uintptr_t ids, uintptr_t out, int num_tokens) {
    ops::embedding_lookup<cfg::HIDDEN>(
        cb(table), reinterpret_cast<const int64_t*>(ids), b(out), (size_t)num_tokens);
}

// final RMSNorm（HIDDEN）
static void rmsnorm(uintptr_t in, uintptr_t out, uintptr_t w, int M) {
    ops::rmsnorm_bf16(cb(in), b(out), cf(w), M, cfg::HIDDEN, cfg::EPS, nullptr);
}

// lm_head: y[M,VOCAB] = x[M,HIDDEN] @ W[VOCAB,HIDDEN]^T  (bf16)
static void lm_head(uintptr_t in, uintptr_t w, uintptr_t out, int M) {
    ops::gemm_bf16(cb(in), cb(w), b(out), M, cfg::VOCAB, cfg::HIDDEN, nullptr);
}

// ===========================================================================
// Mamba block (bf16)
// ===========================================================================
static void mamba_forward(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t in_proj_w, uintptr_t conv1d_w, uintptr_t conv1d_b,
    uintptr_t A_log, uintptr_t D, uintptr_t dt_bias, uintptr_t gnorm_w, uintptr_t out_proj_w,
    uintptr_t out, int B, int S, uintptr_t conv_state_out, uintptr_t ssm_state_out
) {
    Mamba2BlockWeightsBF16 w{ cf(block_norm_w), cb(in_proj_w), cf(conv1d_w), cf(conv1d_b),
        cf(A_log), cf(D), cf(dt_bias), cf(gnorm_w), cb(out_proj_w) };
    mamba_block_forward_bf16<cfg::HIDDEN, cfg::H, cfg::P, cfg::N, cfg::G, cfg::CONV_K>(
        cb(input), w, b(out), B, S, f(conv_state_out), f(ssm_state_out), nullptr);
}

static void mamba_decode(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t in_proj_w, uintptr_t conv1d_w, uintptr_t conv1d_b,
    uintptr_t A_log, uintptr_t D, uintptr_t dt_bias, uintptr_t gnorm_w, uintptr_t out_proj_w,
    uintptr_t conv_state, uintptr_t ssm_state, uintptr_t out, int B
) {
    Mamba2BlockWeightsBF16 w{ cf(block_norm_w), cb(in_proj_w), cf(conv1d_w), cf(conv1d_b),
        cf(A_log), cf(D), cf(dt_bias), cf(gnorm_w), cb(out_proj_w) };
    mamba_block_decode_bf16<cfg::HIDDEN, cfg::H, cfg::P, cfg::N, cfg::G, cfg::CONV_K>(
        cb(input), w, f(conv_state), f(ssm_state), b(out), B, nullptr);
}

// ===========================================================================
// Attention block (bf16)
// ===========================================================================
static void attn_forward(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t q_proj_w, uintptr_t k_proj_w, uintptr_t v_proj_w, uintptr_t o_proj_w,
    uintptr_t out, int S, uintptr_t k_cache, uintptr_t v_cache, int cache_cap
) {
    AttnBlockWeightsBF16 w{ cf(block_norm_w), cb(q_proj_w), cb(k_proj_w), cb(v_proj_w), cb(o_proj_w) };
    attention_block_forward_bf16<cfg::HIDDEN, cfg::H_Q, cfg::H_KV, cfg::HEAD>(
        cb(input), w, b(out), S, b(k_cache), b(v_cache), cache_cap, nullptr);
}

static void attn_decode(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t q_proj_w, uintptr_t k_proj_w, uintptr_t v_proj_w, uintptr_t o_proj_w,
    uintptr_t k_cache, uintptr_t v_cache, int S_cache, int cache_cap, uintptr_t out
) {
    AttnBlockWeightsBF16 w{ cf(block_norm_w), cb(q_proj_w), cb(k_proj_w), cb(v_proj_w), cb(o_proj_w) };
    attention_block_decode_bf16<cfg::HIDDEN, cfg::H_Q, cfg::H_KV, cfg::HEAD>(
        cb(input), w, b(k_cache), b(v_cache), S_cache, cache_cap, b(out), nullptr);
}

// ===========================================================================
// MLP block (bf16)
// ===========================================================================
static void mlp_forward(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t up_proj_w, uintptr_t down_proj_w,
    uintptr_t out, int M
) {
    MLPBlockWeightsBF16 w{ cf(block_norm_w), cb(up_proj_w), cb(down_proj_w) };
    mlp_block_forward_bf16<cfg::HIDDEN, cfg::INTER>(cb(input), w, b(out), M, nullptr);
}

// ===========================================================================
// FP8 变体（included 层；权重 e4m3 + per-row scale，激活设备端动态量化）
//   stored weight_scale 是 per-tensor 标量，Python 侧广播成 [N] per-row 数组即可
//   （gemm_fp8 按列 n 乘 wscale[n]，全相等 = per-tensor）。
// ===========================================================================
static void mamba_forward_fp8(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t in_proj_w, uintptr_t in_proj_wscale,
    uintptr_t conv1d_w, uintptr_t conv1d_b,
    uintptr_t A_log, uintptr_t D, uintptr_t dt_bias, uintptr_t gnorm_w,
    uintptr_t out_proj_w, uintptr_t out_proj_wscale,
    uintptr_t out, int B, int S, uintptr_t conv_state_out, uintptr_t ssm_state_out
) {
    Mamba2BlockWeightsFP8 w{ cf(block_norm_w), c8(in_proj_w), cf(in_proj_wscale),
        cf(conv1d_w), cf(conv1d_b), cf(A_log), cf(D), cf(dt_bias), cf(gnorm_w),
        c8(out_proj_w), cf(out_proj_wscale) };
    mamba_block_forward_fp8<cfg::HIDDEN, cfg::H, cfg::P, cfg::N, cfg::G, cfg::CONV_K>(
        cb(input), w, b(out), B, S, f(conv_state_out), f(ssm_state_out), nullptr);
}

static void mamba_decode_fp8(
    uintptr_t input,
    uintptr_t block_norm_w, uintptr_t in_proj_w, uintptr_t in_proj_wscale,
    uintptr_t conv1d_w, uintptr_t conv1d_b,
    uintptr_t A_log, uintptr_t D, uintptr_t dt_bias, uintptr_t gnorm_w,
    uintptr_t out_proj_w, uintptr_t out_proj_wscale,
    uintptr_t conv_state, uintptr_t ssm_state, uintptr_t out, int B
) {
    Mamba2BlockWeightsFP8 w{ cf(block_norm_w), c8(in_proj_w), cf(in_proj_wscale),
        cf(conv1d_w), cf(conv1d_b), cf(A_log), cf(D), cf(dt_bias), cf(gnorm_w),
        c8(out_proj_w), cf(out_proj_wscale) };
    mamba_block_decode_fp8<cfg::HIDDEN, cfg::H, cfg::P, cfg::N, cfg::G, cfg::CONV_K>(
        cb(input), w, f(conv_state), f(ssm_state), b(out), B, nullptr);
}

static void attn_forward_fp8(
    uintptr_t input, uintptr_t block_norm_w,
    uintptr_t q_w, uintptr_t q_s, uintptr_t k_w, uintptr_t k_s,
    uintptr_t v_w, uintptr_t v_s, uintptr_t o_w, uintptr_t o_s,
    uintptr_t out, int S, uintptr_t k_cache, uintptr_t v_cache, int cache_cap
) {
    AttnBlockWeightsFP8 w{ cf(block_norm_w), c8(q_w), cf(q_s), c8(k_w), cf(k_s),
        c8(v_w), cf(v_s), c8(o_w), cf(o_s) };
    attention_block_forward_fp8<cfg::HIDDEN, cfg::H_Q, cfg::H_KV, cfg::HEAD>(
        cb(input), w, b(out), S, b(k_cache), b(v_cache), cache_cap, nullptr);
}

static void attn_decode_fp8(
    uintptr_t input, uintptr_t block_norm_w,
    uintptr_t q_w, uintptr_t q_s, uintptr_t k_w, uintptr_t k_s,
    uintptr_t v_w, uintptr_t v_s, uintptr_t o_w, uintptr_t o_s,
    uintptr_t k_cache, uintptr_t v_cache, int S_cache, int cache_cap, uintptr_t out
) {
    AttnBlockWeightsFP8 w{ cf(block_norm_w), c8(q_w), cf(q_s), c8(k_w), cf(k_s),
        c8(v_w), cf(v_s), c8(o_w), cf(o_s) };
    attention_block_decode_fp8<cfg::HIDDEN, cfg::H_Q, cfg::H_KV, cfg::HEAD>(
        cb(input), w, b(k_cache), b(v_cache), S_cache, cache_cap, b(out), nullptr);
}

static void mlp_forward_fp8(
    uintptr_t input, uintptr_t block_norm_w,
    uintptr_t up_w, uintptr_t up_s, uintptr_t down_w, uintptr_t down_s,
    uintptr_t out, int M
) {
    MLPBlockWeightsFP8 w{ cf(block_norm_w), c8(up_w), cf(up_s), c8(down_w), cf(down_s) };
    mlp_block_forward_fp8<cfg::HIDDEN, cfg::INTER>(cb(input), w, b(out), M, nullptr);
}

PYBIND11_MODULE(binding, m) {
    m.doc() = "nemotron_infer bf16 whole-model orchestration bindings";

    // config 常量（Python 端取用）
    m.attr("HIDDEN") = cfg::HIDDEN;
    m.attr("H")      = cfg::H;
    m.attr("P")      = cfg::P;
    m.attr("N")      = cfg::N;
    m.attr("G")      = cfg::G;
    m.attr("CONV_K") = cfg::CONV_K;
    m.attr("H_Q")    = cfg::H_Q;
    m.attr("H_KV")   = cfg::H_KV;
    m.attr("HEAD")   = cfg::HEAD;
    m.attr("INTER")  = cfg::INTER;
    m.attr("VOCAB")  = cfg::VOCAB;
    m.attr("CONV_DIM") = cfg::H * cfg::P + 2 * cfg::G * cfg::N;
    m.attr("IN_PROJ")  = cfg::H * cfg::P + (cfg::H * cfg::P + 2 * cfg::G * cfg::N) + cfg::H;

    m.def("reset_allocator", &reset_allocator);
    m.def("free_allocator", &free_allocator);
    m.def("sync", &sync);
    m.def("embedding", &embedding);
    m.def("rmsnorm", &rmsnorm);
    m.def("lm_head", &lm_head);

    m.def("mamba_forward", &mamba_forward);
    m.def("mamba_decode", &mamba_decode);
    m.def("attn_forward", &attn_forward);
    m.def("attn_decode", &attn_decode);
    m.def("mlp_forward", &mlp_forward);

    m.def("mamba_forward_fp8", &mamba_forward_fp8);
    m.def("mamba_decode_fp8", &mamba_decode_fp8);
    m.def("attn_forward_fp8", &attn_forward_fp8);
    m.def("attn_decode_fp8", &attn_decode_fp8);
    m.def("mlp_forward_fp8", &mlp_forward_fp8);
}

// ===========================================================================
// test_attention_block.cu — 单层 Attention NemotronHBlock，对齐 HF 金标准 dump
//   数据：tools/dump_attention_block.py（NoPE GQA causal SDPA + 残差）
//   跑前：uv run python tools/dump_attention_block.py
//
//   HEAD=128（SDPA 核硬约束）；小 config 仅缩 HIDDEN/H/S。
// ===========================================================================
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "model/attention_block.cuh"

using namespace nemotron;
using namespace nemotron::model;
using namespace nemotron::ops;

namespace {

std::string data_dir() {
    const char* e = std::getenv("ATTN_BLOCK_DATA");
    return e ? std::string(e)
             : std::string("D:/project/nemotron_infer/tests/data/attention_block/");
}

bool read_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize(n / sizeof(float));
    return bool(f.read(reinterpret_cast<char*>(out.data()), n));
}

Tensor<float> up(const std::vector<float>& h) {
    auto t = allocate_tensor<float>(TensorShape::make_1d((int64_t)h.size()));
    copy_host_to_device(t, h.data());
    return t;
}

Tensor<__nv_bfloat16> up_bf16(const std::vector<float>& h) {
    std::vector<__nv_bfloat16> hb(h.size());
    for (size_t i = 0; i < h.size(); ++i) hb[i] = __float2bfloat16_rn(h[i]);
    auto t = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d((int64_t)h.size()));
    copy_host_to_device(t, hb.data());
    return t;
}

struct Fp8W { Tensor<__nv_fp8_e4m3> w; Tensor<float> scale; };
Fp8W up_fp8_perrow(const std::vector<float>& src, int Nrows, int K) {
    std::vector<__nv_fp8_e4m3> wq((size_t)Nrows * K);
    std::vector<float> sc(Nrows);
    for (int n = 0; n < Nrows; ++n) {
        float rmax = 0.f;
        for (int k = 0; k < K; ++k) rmax = std::max(rmax, std::abs(src[(size_t)n * K + k]));
        float s = rmax / 448.f; if (s < 1e-10f) s = 1.f / 448.f;
        sc[n] = s; float inv = 1.f / s;
        for (int k = 0; k < K; ++k) {
            float q = src[(size_t)n * K + k] * inv;
            q = std::max(-448.f, std::min(448.f, q));
            wq[(size_t)n * K + k] = static_cast<__nv_fp8_e4m3>(q);
        }
    }
    Fp8W r;
    r.w = allocate_tensor<__nv_fp8_e4m3>(TensorShape::make_1d((int64_t)wq.size()));
    r.scale = allocate_tensor<float>(TensorShape::make_1d(Nrows));
    copy_host_to_device(r.w, wq.data());
    copy_host_to_device(r.scale, sc.data());
    return r;
}

constexpr int HIDDEN = 256, H_Q = 4, H_KV = 2, HEAD = 128, S = 40, B = 1;  // S>32 覆盖 SDPA 多 tile
constexpr int QD = H_Q * HEAD, KD = H_KV * HEAD;

struct AttnData {
    std::vector<float> input, bnw, qw, kw, vw, ow, expected;
    bool ok = false;
};
AttnData load() {
    AttnData a;
    const std::string d = data_dir();
    if (!read_bin(d + "input.bin", a.input)) return a;
    read_bin(d + "block_norm_w.bin", a.bnw);
    read_bin(d + "q_proj_w.bin", a.qw);
    read_bin(d + "k_proj_w.bin", a.kw);
    read_bin(d + "v_proj_w.bin", a.vw);
    read_bin(d + "o_proj_w.bin", a.ow);
    read_bin(d + "expected.bin", a.expected);
    a.ok = true;
    return a;
}

}  // namespace

class AttnBlockTest : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); }
};

TEST_F(AttnBlockTest, FP32_MatchHF) {
    auto a = load();
    if (!a.ok) GTEST_SKIP() << "缺 dump：uv run python tools/dump_attention_block.py";

    auto d_in = up(a.input), d_bnw = up(a.bnw);
    auto d_qw = up(a.qw), d_kw = up(a.kw), d_vw = up(a.vw), d_ow = up(a.ow);
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    AttnBlockWeightsFP32 w{ d_bnw.data_, d_qw.data_, d_kw.data_, d_vw.data_, d_ow.data_ };
    attention_block_forward_fp32<HIDDEN, H_Q, H_KV, HEAD>(d_in.data_, w, d_out.data_, S);
    cudaDeviceSynchronize();

    std::vector<float> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    ASSERT_EQ(a.expected.size(), got.size());
    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, (double)std::abs(got[i] - a.expected[i]));
        EXPECT_NEAR(got[i], a.expected[i], 2e-3f) << " i=" << i;
    }
    printf("  [AttnBlock-fp32] max_abs_err vs HF = %.3e  (%zu elems)\n", max_err, got.size());
}

TEST_F(AttnBlockTest, BF16_MatchHF) {
    auto a = load();
    if (!a.ok) GTEST_SKIP() << "缺 dump";

    auto d_in = up_bf16(a.input);
    auto d_bnw = up(a.bnw);
    auto d_qw = up_bf16(a.qw), d_kw = up_bf16(a.kw), d_vw = up_bf16(a.vw), d_ow = up_bf16(a.ow);
    auto d_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    AttnBlockWeightsBF16 w{ d_bnw.data_, d_qw.data_, d_kw.data_, d_vw.data_, d_ow.data_ };
    attention_block_forward_bf16<HIDDEN, H_Q, H_KV, HEAD>(d_in.data_, w, d_out.data_, S);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = __bfloat162float(got[i]);
        max_err = std::max(max_err, (double)std::abs(g - a.expected[i]));
        EXPECT_NEAR(g, a.expected[i], 2e-1f) << " i=" << i;
    }
    printf("  [AttnBlock-bf16] max_abs_err vs HF(fp32) = %.3e\n", max_err);
}

TEST_F(AttnBlockTest, FP8_MatchHF) {
    auto a = load();
    if (!a.ok) GTEST_SKIP() << "缺 dump";

    auto d_in = up_bf16(a.input);
    auto d_bnw = up(a.bnw);
    auto qw = up_fp8_perrow(a.qw, QD, HIDDEN);
    auto kw = up_fp8_perrow(a.kw, KD, HIDDEN);
    auto vw = up_fp8_perrow(a.vw, KD, HIDDEN);
    auto ow = up_fp8_perrow(a.ow, HIDDEN, QD);
    auto d_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    AttnBlockWeightsFP8 w{ d_bnw.data_,
        qw.w.data_, qw.scale.data_, kw.w.data_, kw.scale.data_,
        vw.w.data_, vw.scale.data_, ow.w.data_, ow.scale.data_ };
    attention_block_forward_fp8<HIDDEN, H_Q, H_KV, HEAD>(d_in.data_, w, d_out.data_, S);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    double max_err = 0.0, sse = 0.0, ssr = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = __bfloat162float(got[i]);
        max_err = std::max(max_err, (double)std::abs(g - a.expected[i]));
        sse += (double)(g - a.expected[i]) * (g - a.expected[i]);
        ssr += (double)a.expected[i] * a.expected[i];
    }
    double rel = std::sqrt(sse / std::max(ssr, 1e-12));
    printf("  [AttnBlock-fp8]  max_abs_err = %.3e  rel_l2 = %.3e\n", max_err, rel);
    EXPECT_LT(rel, 0.3);
}

// decode 续接：prefill 前 S-1 token 填 KV cache，decode 第 S 个 token，
// 对齐 HF 全序列 expected 末行（验证 KV cache 写入 + sdpa_decode 追加正确）。
TEST_F(AttnBlockTest, BF16_DecodeMatchHF) {
    auto a = load();
    if (!a.ok) GTEST_SKIP() << "缺 dump";
    using bf16 = __nv_bfloat16;

    auto d_in = up_bf16(a.input);
    auto d_bnw = up(a.bnw);
    auto d_qw = up_bf16(a.qw), d_kw = up_bf16(a.kw), d_vw = up_bf16(a.vw), d_ow = up_bf16(a.ow);
    AttnBlockWeightsBF16 w{ d_bnw.data_, d_qw.data_, d_kw.data_, d_vw.data_, d_ow.data_ };

    // KV cache 容量 = S（留出 decode token 的位置 S-1）
    auto k_cache = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)H_KV * S * HEAD));
    auto v_cache = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)H_KV * S * HEAD));
    const int Spre = S - 1;
    auto pre_out = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)Spre * HIDDEN));
    auto dec_out = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)HIDDEN));
    cudaDeviceSynchronize();

    // prefill 前 S-1 token，写 KV cache[0,S-1)
    attention_block_forward_bf16<HIDDEN, H_Q, H_KV, HEAD>(
        d_in.data_, w, pre_out.data_, Spre, k_cache.data_, v_cache.data_, /*cache_cap*/S);
    cudaDeviceSynchronize();

    // decode 第 S 个 token（续接），S_cache=Spre，cache_cap=S
    const bf16* tok = d_in.data_ + (size_t)(S - 1) * HIDDEN;
    attention_block_decode_bf16<HIDDEN, H_Q, H_KV, HEAD>(
        tok, w, k_cache.data_, v_cache.data_, /*S_cache*/Spre, /*cache_cap*/S, dec_out.data_);
    cudaDeviceSynchronize();

    std::vector<bf16> got(HIDDEN);
    copy_device_to_host(got.data(), dec_out);
    cudaDeviceSynchronize();

    // bf16 逐点误差随 S 增长（注意力累积更多位置），用尺度无关的 rel_l2 判定更稳健
    double max_err = 0.0, sse = 0.0, ssr = 0.0;
    for (int i = 0; i < HIDDEN; ++i) {
        float g = __bfloat162float(got[i]);
        float e = a.expected[(size_t)(S - 1) * HIDDEN + i];
        max_err = std::max(max_err, (double)std::abs(g - e));
        sse += (double)(g - e) * (g - e);
        ssr += (double)e * e;
    }
    double rel = std::sqrt(sse / std::max(ssr, 1e-12));
    printf("  [AttnBlock-decode] token[%d] max_abs_err = %.3e  rel_l2 = %.3e\n", S - 1, max_err, rel);
    EXPECT_LT(rel, 0.1);
}

// fp8 decode 续接（KV cache 仍 bf16；q/k/v/o_proj 走 fp8）
TEST_F(AttnBlockTest, FP8_DecodeMatchHF) {
    auto a = load();
    if (!a.ok) GTEST_SKIP() << "缺 dump";
    using bf16 = __nv_bfloat16;

    auto d_in = up_bf16(a.input);
    auto d_bnw = up(a.bnw);
    auto qw = up_fp8_perrow(a.qw, QD, HIDDEN);
    auto kw = up_fp8_perrow(a.kw, KD, HIDDEN);
    auto vw = up_fp8_perrow(a.vw, KD, HIDDEN);
    auto ow = up_fp8_perrow(a.ow, HIDDEN, QD);
    AttnBlockWeightsFP8 w{ d_bnw.data_,
        qw.w.data_, qw.scale.data_, kw.w.data_, kw.scale.data_,
        vw.w.data_, vw.scale.data_, ow.w.data_, ow.scale.data_ };

    auto k_cache = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)H_KV * S * HEAD));
    auto v_cache = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)H_KV * S * HEAD));
    const int Spre = S - 1;
    auto pre_out = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)Spre * HIDDEN));
    auto dec_out = allocate_tensor_zeros<bf16>(TensorShape::make_1d((int64_t)HIDDEN));
    cudaDeviceSynchronize();

    attention_block_forward_fp8<HIDDEN, H_Q, H_KV, HEAD>(
        d_in.data_, w, pre_out.data_, Spre, k_cache.data_, v_cache.data_, S);
    cudaDeviceSynchronize();

    const bf16* tok = d_in.data_ + (size_t)(S - 1) * HIDDEN;
    attention_block_decode_fp8<HIDDEN, H_Q, H_KV, HEAD>(
        tok, w, k_cache.data_, v_cache.data_, Spre, S, dec_out.data_);
    cudaDeviceSynchronize();

    std::vector<bf16> got(HIDDEN);
    copy_device_to_host(got.data(), dec_out);
    cudaDeviceSynchronize();

    double max_err = 0.0, sse = 0.0, ssr = 0.0;
    for (int i = 0; i < HIDDEN; ++i) {
        float g = __bfloat162float(got[i]);
        float e = a.expected[(size_t)(S - 1) * HIDDEN + i];
        max_err = std::max(max_err, (double)std::abs(g - e));
        sse += (double)(g - e) * (g - e);
        ssr += (double)e * e;
    }
    double rel = std::sqrt(sse / std::max(ssr, 1e-12));
    printf("  [AttnBlock-decode-fp8] token[%d] max_abs_err = %.3e  rel_l2 = %.3e\n",
           S - 1, max_err, rel);
    EXPECT_LT(rel, 0.35);
}

// ===========================================================================
// test_mlp_block.cu — 单层 MLP NemotronHBlock，对齐 HF 金标准 dump
//   数据：tools/dump_mlp_block.py（NemotronHMLP = down(relu²(up(x))) + 残差）
//   跑前：uv run python tools/dump_mlp_block.py
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
#include "model/mlp_block.cuh"

using namespace nemotron;
using namespace nemotron::model;
using namespace nemotron::ops;

namespace {

std::string data_dir() {
    const char* e = std::getenv("MLP_BLOCK_DATA");
    return e ? std::string(e)
             : std::string("D:/project/nemotron_infer/tests/data/mlp_block/");
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

constexpr int HIDDEN = 64, INTER = 256, S = 8, B = 1;

struct MLPData {
    std::vector<float> input, bnw, upw, dnw, expected;
    bool ok = false;
};
MLPData load() {
    MLPData m;
    const std::string d = data_dir();
    if (!read_bin(d + "input.bin", m.input)) return m;
    read_bin(d + "block_norm_w.bin", m.bnw);
    read_bin(d + "up_proj_w.bin", m.upw);
    read_bin(d + "down_proj_w.bin", m.dnw);
    read_bin(d + "expected.bin", m.expected);
    m.ok = true;
    return m;
}

}  // namespace

class MLPBlockTest : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); }
};

TEST_F(MLPBlockTest, FP32_MatchHF) {
    auto m = load();
    if (!m.ok) GTEST_SKIP() << "缺 dump：uv run python tools/dump_mlp_block.py";

    auto d_in = up(m.input), d_bnw = up(m.bnw), d_upw = up(m.upw), d_dnw = up(m.dnw);
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    MLPBlockWeightsFP32 w{ d_bnw.data_, d_upw.data_, d_dnw.data_ };
    mlp_block_forward_fp32<HIDDEN, INTER>(d_in.data_, w, d_out.data_, B * S);
    cudaDeviceSynchronize();

    std::vector<float> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    ASSERT_EQ(m.expected.size(), got.size());
    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, (double)std::abs(got[i] - m.expected[i]));
        EXPECT_NEAR(got[i], m.expected[i], 1e-3f) << " i=" << i;
    }
    printf("  [MLPBlock-fp32] max_abs_err vs HF = %.3e  (%zu elems)\n", max_err, got.size());
}

TEST_F(MLPBlockTest, BF16_MatchHF) {
    auto m = load();
    if (!m.ok) GTEST_SKIP() << "缺 dump";

    auto d_in = up_bf16(m.input), d_upw = up_bf16(m.upw), d_dnw = up_bf16(m.dnw);
    auto d_bnw = up(m.bnw);
    auto d_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    MLPBlockWeightsBF16 w{ d_bnw.data_, d_upw.data_, d_dnw.data_ };
    mlp_block_forward_bf16<HIDDEN, INTER>(d_in.data_, w, d_out.data_, B * S);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = __bfloat162float(got[i]);
        max_err = std::max(max_err, (double)std::abs(g - m.expected[i]));
        EXPECT_NEAR(g, m.expected[i], 1.5e-1f) << " i=" << i;
    }
    printf("  [MLPBlock-bf16] max_abs_err vs HF(fp32) = %.3e\n", max_err);
}

TEST_F(MLPBlockTest, FP8_MatchHF) {
    auto m = load();
    if (!m.ok) GTEST_SKIP() << "缺 dump";

    auto d_in = up_bf16(m.input);
    auto d_bnw = up(m.bnw);
    auto upw = up_fp8_perrow(m.upw, INTER, HIDDEN);
    auto dnw = up_fp8_perrow(m.dnw, HIDDEN, INTER);
    auto d_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    MLPBlockWeightsFP8 w{ d_bnw.data_, upw.w.data_, upw.scale.data_,
                          dnw.w.data_, dnw.scale.data_ };
    mlp_block_forward_fp8<HIDDEN, INTER>(d_in.data_, w, d_out.data_, B * S);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    double max_err = 0.0, sse = 0.0, ssr = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = __bfloat162float(got[i]);
        max_err = std::max(max_err, (double)std::abs(g - m.expected[i]));
        sse += (double)(g - m.expected[i]) * (g - m.expected[i]);
        ssr += (double)m.expected[i] * m.expected[i];
    }
    double rel = std::sqrt(sse / std::max(ssr, 1e-12));
    printf("  [MLPBlock-fp8]  max_abs_err = %.3e  rel_l2 = %.3e\n", max_err, rel);
    EXPECT_LT(rel, 0.3);
}

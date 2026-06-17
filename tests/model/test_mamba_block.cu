// ===========================================================================
// test_mamba_block.cu — 单层 Mamba NemotronHBlock 组装，对齐 HF 金标准 dump
//
// 数据由 tools/dump_mamba_block.py 生成（HF chunked SSD 数学 verbatim）。
// 跑前先：uv run python tools/dump_mamba_block.py
// ===========================================================================
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "model/mamba_block.cuh"

using namespace nemotron;
using namespace nemotron::model;
using namespace nemotron::ops;
using namespace nemotron::ops::mamba2;

namespace {

std::string data_dir() {
    const char* e = std::getenv("MAMBA_BLOCK_DATA");
    return e ? std::string(e)
             : std::string("D:/project/nemotron_infer/tests/data/mamba_block/");
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

// fp32 host 向量 → bf16 device 张量（dump 是 fp32 金标准，bf16 路径在此转换）
Tensor<__nv_bfloat16> up_bf16(const std::vector<float>& h) {
    std::vector<__nv_bfloat16> hb(h.size());
    for (size_t i = 0; i < h.size(); ++i) hb[i] = __float2bfloat16_rn(h[i]);
    auto t = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d((int64_t)h.size()));
    copy_host_to_device(t, hb.data());
    return t;
}

// fp32 权重 [Nrows,K] 行主序 → e4m3 + per-row(输出通道) scale（host 量化，对齐 gemm_fp8）
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

}  // namespace

class MambaBlockTest : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); }
};

TEST_F(MambaBlockTest, FP32_MatchHF) {
    // 与 dump 脚本一致的小 config
    constexpr int HIDDEN = 64, H = 4, P = 16, N = 16, G = 2, S = 24, B = 1;
    const std::string d = data_dir();

    std::vector<float> input, bnw, inw, cw, cb, alog, Dp, dtb, gnw, opw, expected;
    if (!read_bin(d + "input.bin", input)) {
        GTEST_SKIP() << "缺 dump 数据，先跑: uv run python tools/dump_mamba_block.py";
    }
    ASSERT_TRUE(read_bin(d + "block_norm_w.bin", bnw));
    ASSERT_TRUE(read_bin(d + "in_proj_w.bin", inw));
    ASSERT_TRUE(read_bin(d + "conv1d_w.bin", cw));
    ASSERT_TRUE(read_bin(d + "conv1d_b.bin", cb));
    ASSERT_TRUE(read_bin(d + "A_log.bin", alog));
    ASSERT_TRUE(read_bin(d + "D.bin", Dp));
    ASSERT_TRUE(read_bin(d + "dt_bias.bin", dtb));
    ASSERT_TRUE(read_bin(d + "gnorm_w.bin", gnw));
    ASSERT_TRUE(read_bin(d + "out_proj_w.bin", opw));
    ASSERT_TRUE(read_bin(d + "expected.bin", expected));

    auto d_in  = up(input);
    auto d_bnw = up(bnw);  auto d_inw = up(inw);  auto d_cw = up(cw);  auto d_cb = up(cb);
    auto d_alog = up(alog); auto d_D = up(Dp);    auto d_dtb = up(dtb);
    auto d_gnw = up(gnw);  auto d_opw = up(opw);
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    Mamba2BlockWeightsFP32 w{
        d_bnw.data_, d_inw.data_, d_cw.data_, d_cb.data_,
        d_alog.data_, d_D.data_, d_dtb.data_, d_gnw.data_, d_opw.data_
    };

    mamba_block_forward_fp32<HIDDEN, H, P, N, G>(d_in.data_, w, d_out.data_, B, S);
    cudaDeviceSynchronize();

    std::vector<float> got(B * S * HIDDEN);
    copy_device_to_host(got.data(), d_out);
    cudaDeviceSynchronize();

    ASSERT_EQ(expected.size(), got.size());
    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, (double)std::abs(got[i] - expected[i]));
        EXPECT_NEAR(got[i], expected[i], 1e-3f) << " i=" << i;
    }
    printf("  [MambaBlock] max_abs_err vs HF = %.3e  (%zu elems)\n", max_err, got.size());
}

// bf16 混合精度路径对齐 HF（fp32 金标准），bf16 容差
TEST_F(MambaBlockTest, BF16_MatchHF) {
    constexpr int HIDDEN = 64, H = 4, P = 16, N = 16, G = 2, S = 24, B = 1;
    const std::string d = data_dir();

    std::vector<float> input, bnw, inw, cw, cb, alog, Dp, dtb, gnw, opw, expected;
    if (!read_bin(d + "input.bin", input)) {
        GTEST_SKIP() << "缺 dump 数据，先跑: uv run python tools/dump_mamba_block.py";
    }
    ASSERT_TRUE(read_bin(d + "block_norm_w.bin", bnw));
    ASSERT_TRUE(read_bin(d + "in_proj_w.bin", inw));
    ASSERT_TRUE(read_bin(d + "conv1d_w.bin", cw));
    ASSERT_TRUE(read_bin(d + "conv1d_b.bin", cb));
    ASSERT_TRUE(read_bin(d + "A_log.bin", alog));
    ASSERT_TRUE(read_bin(d + "D.bin", Dp));
    ASSERT_TRUE(read_bin(d + "dt_bias.bin", dtb));
    ASSERT_TRUE(read_bin(d + "gnorm_w.bin", gnw));
    ASSERT_TRUE(read_bin(d + "out_proj_w.bin", opw));
    ASSERT_TRUE(read_bin(d + "expected.bin", expected));

    auto d_in  = up_bf16(input);
    auto d_bnw = up(bnw);  auto d_inw = up_bf16(inw); auto d_cw = up(cw);  auto d_cb = up(cb);
    auto d_alog = up(alog); auto d_D = up(Dp);        auto d_dtb = up(dtb);
    auto d_gnw = up(gnw);  auto d_opw = up_bf16(opw);
    auto d_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    Mamba2BlockWeightsBF16 w{
        d_bnw.data_, d_inw.data_, d_cw.data_, d_cb.data_,
        d_alog.data_, d_D.data_, d_dtb.data_, d_gnw.data_, d_opw.data_
    };

    mamba_block_forward_bf16<HIDDEN, H, P, N, G>(d_in.data_, w, d_out.data_, B, S);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got_b(B * S * HIDDEN);
    copy_device_to_host(got_b.data(), d_out);
    cudaDeviceSynchronize();

    ASSERT_EQ(expected.size(), got_b.size());
    double max_err = 0.0;
    for (size_t i = 0; i < got_b.size(); ++i) {
        float g = __bfloat162float(got_b[i]);
        max_err = std::max(max_err, (double)std::abs(g - expected[i]));
        EXPECT_NEAR(g, expected[i], 1.5e-1f) << " i=" << i;
    }
    printf("  [MambaBlock-bf16] max_abs_err vs HF(fp32) = %.3e  (%zu elems)\n",
           max_err, got_b.size());
}

// fp8 GEMM 混合精度路径对齐 HF（fp32 金标准），fp8 容差（最宽）
TEST_F(MambaBlockTest, FP8_MatchHF) {
    constexpr int HIDDEN = 64, H = 4, P = 16, N = 16, G = 2, S = 24, B = 1;
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N, PROJ = INTER + CONV_DIM + H;
    const std::string d = data_dir();

    std::vector<float> input, bnw, inw, cw, cb, alog, Dp, dtb, gnw, opw, expected;
    if (!read_bin(d + "input.bin", input)) {
        GTEST_SKIP() << "缺 dump 数据，先跑: uv run python tools/dump_mamba_block.py";
    }
    ASSERT_TRUE(read_bin(d + "block_norm_w.bin", bnw));
    ASSERT_TRUE(read_bin(d + "in_proj_w.bin", inw));
    ASSERT_TRUE(read_bin(d + "conv1d_w.bin", cw));
    ASSERT_TRUE(read_bin(d + "conv1d_b.bin", cb));
    ASSERT_TRUE(read_bin(d + "A_log.bin", alog));
    ASSERT_TRUE(read_bin(d + "D.bin", Dp));
    ASSERT_TRUE(read_bin(d + "dt_bias.bin", dtb));
    ASSERT_TRUE(read_bin(d + "gnorm_w.bin", gnw));
    ASSERT_TRUE(read_bin(d + "out_proj_w.bin", opw));
    ASSERT_TRUE(read_bin(d + "expected.bin", expected));

    auto d_in  = up_bf16(input);
    auto d_bnw = up(bnw);  auto d_cw = up(cw);  auto d_cb = up(cb);
    auto d_alog = up(alog); auto d_D = up(Dp);  auto d_dtb = up(dtb);  auto d_gnw = up(gnw);
    auto inproj = up_fp8_perrow(inw, PROJ, HIDDEN);
    auto outproj = up_fp8_perrow(opw, HIDDEN, INTER);
    auto d_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)(B * S * HIDDEN)));
    cudaDeviceSynchronize();

    Mamba2BlockWeightsFP8 w{
        d_bnw.data_, inproj.w.data_, inproj.scale.data_, d_cw.data_, d_cb.data_,
        d_alog.data_, d_D.data_, d_dtb.data_, d_gnw.data_,
        outproj.w.data_, outproj.scale.data_
    };

    mamba_block_forward_fp8<HIDDEN, H, P, N, G>(d_in.data_, w, d_out.data_, B, S);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got_b(B * S * HIDDEN);
    copy_device_to_host(got_b.data(), d_out);
    cudaDeviceSynchronize();

    ASSERT_EQ(expected.size(), got_b.size());
    double max_err = 0.0, sse = 0.0, ssr = 0.0;
    for (size_t i = 0; i < got_b.size(); ++i) {
        float g = __bfloat162float(got_b[i]);
        max_err = std::max(max_err, (double)std::abs(g - expected[i]));
        sse += (double)(g - expected[i]) * (g - expected[i]);
        ssr += (double)expected[i] * expected[i];
    }
    printf("  [MambaBlock-fp8]  max_abs_err vs HF(fp32) = %.3e  rel_l2 = %.3e  (%zu elems)\n",
           max_err, std::sqrt(sse / std::max(ssr, 1e-12)), got_b.size());
    EXPECT_LT(std::sqrt(sse / std::max(ssr, 1e-12)), 0.3);  // 相对 L2 < 30%
}

// decode 续接验证：prefill 前 S-1 token 产出末态，decode 第 S 个 token，
// 其输出应等于 HF 全序列 expected 的最后一行（验证 prefill→decode 状态交接正确）。
// 复用既有 dump（无需扩展脚本）：expected[S-1] 就是位置 S-1 的模型输出。
TEST_F(MambaBlockTest, BF16_DecodeMatchHF) {
    constexpr int HIDDEN = 64, H = 4, P = 16, N = 16, G = 2, S = 24, B = 1;
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N, CONV_K = 4;
    const std::string d = data_dir();

    std::vector<float> input, bnw, inw, cw, cb, alog, Dp, dtb, gnw, opw, expected;
    if (!read_bin(d + "input.bin", input)) {
        GTEST_SKIP() << "缺 dump 数据，先跑: uv run python tools/dump_mamba_block.py";
    }
    ASSERT_TRUE(read_bin(d + "block_norm_w.bin", bnw));
    ASSERT_TRUE(read_bin(d + "in_proj_w.bin", inw));
    ASSERT_TRUE(read_bin(d + "conv1d_w.bin", cw));
    ASSERT_TRUE(read_bin(d + "conv1d_b.bin", cb));
    ASSERT_TRUE(read_bin(d + "A_log.bin", alog));
    ASSERT_TRUE(read_bin(d + "D.bin", Dp));
    ASSERT_TRUE(read_bin(d + "dt_bias.bin", dtb));
    ASSERT_TRUE(read_bin(d + "gnorm_w.bin", gnw));
    ASSERT_TRUE(read_bin(d + "out_proj_w.bin", opw));
    ASSERT_TRUE(read_bin(d + "expected.bin", expected));

    auto d_in  = up_bf16(input);                     // [S, HIDDEN]
    auto d_bnw = up(bnw);  auto d_inw = up_bf16(inw); auto d_cw = up(cw);  auto d_cb = up(cb);
    auto d_alog = up(alog); auto d_D = up(Dp);        auto d_dtb = up(dtb);
    auto d_gnw = up(gnw);  auto d_opw = up_bf16(opw);

    Mamba2BlockWeightsBF16 w{
        d_bnw.data_, d_inw.data_, d_cw.data_, d_cb.data_,
        d_alog.data_, d_D.data_, d_dtb.data_, d_gnw.data_, d_opw.data_
    };

    // 末态缓冲
    auto conv_state = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B * CONV_DIM * (CONV_K - 1)));
    auto ssm_state  = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B * H * P * N));
    const int Spre = S - 1;
    auto pre_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)Spre * HIDDEN));
    auto dec_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)HIDDEN));
    cudaDeviceSynchronize();

    // prefill 前 S-1 个 token，产出 conv_state + ssm_state
    mamba_block_forward_bf16<HIDDEN, H, P, N, G>(
        d_in.data_, w, pre_out.data_, B, Spre, conv_state.data_, ssm_state.data_);
    cudaDeviceSynchronize();

    // decode 第 S 个 token（input 最后一行），续接末态
    const __nv_bfloat16* tok = d_in.data_ + (size_t)(S - 1) * HIDDEN;
    mamba_block_decode_bf16<HIDDEN, H, P, N, G>(
        tok, w, conv_state.data_, ssm_state.data_, dec_out.data_, B);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(HIDDEN);
    copy_device_to_host(got.data(), dec_out);
    cudaDeviceSynchronize();

    double max_err = 0.0;
    for (int i = 0; i < HIDDEN; ++i) {
        float g = __bfloat162float(got[i]);
        float e = expected[(size_t)(S - 1) * HIDDEN + i];
        max_err = std::max(max_err, (double)std::abs(g - e));
        EXPECT_NEAR(g, e, 1.5e-1f) << " i=" << i;
    }
    printf("  [MambaBlock-decode] token[%d] max_abs_err vs HF = %.3e\n", S - 1, max_err);
}

// fp8 decode 续接验证（全 fp8 路径：fp8 prefill 产末态 → fp8 decode → 对 HF）
TEST_F(MambaBlockTest, FP8_DecodeMatchHF) {
    constexpr int HIDDEN = 64, H = 4, P = 16, N = 16, G = 2, S = 24, B = 1;
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N, PROJ = INTER + CONV_DIM + H, CONV_K = 4;
    const std::string d = data_dir();

    std::vector<float> input, bnw, inw, cw, cb, alog, Dp, dtb, gnw, opw, expected;
    if (!read_bin(d + "input.bin", input)) GTEST_SKIP() << "先跑 dump 脚本";
    ASSERT_TRUE(read_bin(d + "block_norm_w.bin", bnw));
    ASSERT_TRUE(read_bin(d + "in_proj_w.bin", inw));
    ASSERT_TRUE(read_bin(d + "conv1d_w.bin", cw));
    ASSERT_TRUE(read_bin(d + "conv1d_b.bin", cb));
    ASSERT_TRUE(read_bin(d + "A_log.bin", alog));
    ASSERT_TRUE(read_bin(d + "D.bin", Dp));
    ASSERT_TRUE(read_bin(d + "dt_bias.bin", dtb));
    ASSERT_TRUE(read_bin(d + "gnorm_w.bin", gnw));
    ASSERT_TRUE(read_bin(d + "out_proj_w.bin", opw));
    ASSERT_TRUE(read_bin(d + "expected.bin", expected));

    auto d_in = up_bf16(input);
    auto d_bnw = up(bnw); auto d_cw = up(cw); auto d_cb = up(cb);
    auto d_alog = up(alog); auto d_D = up(Dp); auto d_dtb = up(dtb); auto d_gnw = up(gnw);
    auto inproj = up_fp8_perrow(inw, PROJ, HIDDEN);
    auto outproj = up_fp8_perrow(opw, HIDDEN, INTER);

    Mamba2BlockWeightsFP8 w{
        d_bnw.data_, inproj.w.data_, inproj.scale.data_, d_cw.data_, d_cb.data_,
        d_alog.data_, d_D.data_, d_dtb.data_, d_gnw.data_, outproj.w.data_, outproj.scale.data_
    };

    auto conv_state = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B * CONV_DIM * (CONV_K - 1)));
    auto ssm_state  = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B * H * P * N));
    const int Spre = S - 1;
    auto pre_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)Spre * HIDDEN));
    auto dec_out = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d((int64_t)HIDDEN));
    cudaDeviceSynchronize();

    mamba_block_forward_fp8<HIDDEN, H, P, N, G>(
        d_in.data_, w, pre_out.data_, B, Spre, conv_state.data_, ssm_state.data_);
    cudaDeviceSynchronize();
    const __nv_bfloat16* tok = d_in.data_ + (size_t)(S - 1) * HIDDEN;
    mamba_block_decode_fp8<HIDDEN, H, P, N, G>(
        tok, w, conv_state.data_, ssm_state.data_, dec_out.data_, B);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(HIDDEN);
    copy_device_to_host(got.data(), dec_out);
    cudaDeviceSynchronize();

    double max_err = 0.0, sse = 0.0, ssr = 0.0;
    for (int i = 0; i < HIDDEN; ++i) {
        float g = __bfloat162float(got[i]);
        float e = expected[(size_t)(S - 1) * HIDDEN + i];
        max_err = std::max(max_err, (double)std::abs(g - e));
        sse += (double)(g - e) * (g - e); ssr += (double)e * e;
    }
    printf("  [MambaBlock-decode-fp8] token[%d] max_err=%.3e rel_l2=%.3e\n",
           S - 1, max_err, std::sqrt(sse / std::max(ssr, 1e-12)));
    EXPECT_LT(std::sqrt(sse / std::max(ssr, 1e-12)), 0.35);  // 精度不崩：相对 L2 < 35%
}

// 分阶段对照：逐 op 比对中间张量，定位首个发散的阶段
TEST_F(MambaBlockTest, StageDebug) {
    constexpr int HIDDEN = 64, H = 4, P = 16, N = 16, G = 2, S = 24, B = 1;
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N, GROUP_SIZE = INTER / G;
    const int M = B * S;
    const std::string d = data_dir();

    std::vector<float> input, bnw, inw, cw, cb, alog, Dp, dtb, gnw, opw;
    std::vector<float> normed_h, gate_h, xbc_h, dt_h, xbcc_h, scany_h, gnormed_h, mixer_h;
    if (!read_bin(d + "input.bin", input)) GTEST_SKIP() << "先跑 dump 脚本";
    read_bin(d + "block_norm_w.bin", bnw); read_bin(d + "in_proj_w.bin", inw);
    read_bin(d + "conv1d_w.bin", cw); read_bin(d + "conv1d_b.bin", cb);
    read_bin(d + "A_log.bin", alog); read_bin(d + "D.bin", Dp); read_bin(d + "dt_bias.bin", dtb);
    read_bin(d + "gnorm_w.bin", gnw); read_bin(d + "out_proj_w.bin", opw);
    read_bin(d + "normed.bin", normed_h); read_bin(d + "gate.bin", gate_h);
    read_bin(d + "xbc.bin", xbc_h); read_bin(d + "dt_raw.bin", dt_h);
    read_bin(d + "xbc_conv.bin", xbcc_h); read_bin(d + "scan_y.bin", scany_h);
    read_bin(d + "gnormed.bin", gnormed_h); read_bin(d + "mixer_out.bin", mixer_h);

    auto d_in = up(input), d_bnw = up(bnw), d_inw = up(inw), d_cw = up(cw), d_cb = up(cb);
    auto d_alog = up(alog), d_D = up(Dp), d_dtb = up(dtb), d_gnw = up(gnw), d_opw = up(opw);
    cudaDeviceSynchronize();

    auto check = [](const char* name, const float* dptr, const std::vector<float>& exp) {
        std::vector<float> got(exp.size());
        cudaMemcpy(got.data(), dptr, exp.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double me = 0; size_t worst = 0;
        for (size_t i = 0; i < exp.size(); ++i) {
            double e = std::abs((double)got[i] - exp[i]);
            if (e > me) { me = e; worst = i; }
        }
        printf("  [stage] %-9s max_err=%.3e  got[%zu]=%.5f exp=%.5f\n",
               name, me, worst, got[worst], exp[worst]);
        return me;
    };

    auto normed   = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));
    auto gate     = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto xbc      = allocate_tensor<float>(TensorShape::make_2d(M, CONV_DIM));
    auto dt       = allocate_tensor<float>(TensorShape::make_2d(M, H));
    auto xbc_conv = allocate_tensor<float>(TensorShape::make_2d(M, CONV_DIM));
    auto x_buf    = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto B_buf    = allocate_tensor<float>(TensorShape::make_2d(M, G * N));
    auto C_buf    = allocate_tensor<float>(TensorShape::make_2d(M, G * N));
    auto y_buf    = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto gnormed  = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto mixer    = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));

    rmsnorm_fp32(d_in.data_, normed.data_, d_bnw.data_, M, HIDDEN, 1e-5f);
    cudaDeviceSynchronize(); check("normed", normed.data_, normed_h);

    gemm_fp32(normed.data_, d_inw.data_, gate.data_, M, INTER, HIDDEN, true);
    gemm_fp32(normed.data_, d_inw.data_ + (size_t)INTER * HIDDEN, xbc.data_, M, CONV_DIM, HIDDEN, true);
    gemm_fp32(normed.data_, d_inw.data_ + (size_t)(INTER + CONV_DIM) * HIDDEN, dt.data_, M, H, HIDDEN, true);
    cudaDeviceSynchronize();
    check("gate", gate.data_, gate_h);
    check("xbc", xbc.data_, xbc_h);
    check("dt_raw", dt.data_, dt_h);

    causal_conv1d_prefill_fp32<64>(xbc.data_, d_cw.data_, d_cb.data_, xbc_conv.data_, nullptr, CONV_DIM, S, B);
    cudaDeviceSynchronize(); check("xbc_conv", xbc_conv.data_, xbcc_h);

    cudaMemcpy2D(x_buf.data_, (size_t)INTER * 4, xbc_conv.data_, (size_t)CONV_DIM * 4, (size_t)INTER * 4, M, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(B_buf.data_, (size_t)(G * N) * 4, xbc_conv.data_ + INTER, (size_t)CONV_DIM * 4, (size_t)(G * N) * 4, M, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(C_buf.data_, (size_t)(G * N) * 4, xbc_conv.data_ + INTER + G * N, (size_t)CONV_DIM * 4, (size_t)(G * N) * 4, M, cudaMemcpyDeviceToDevice);

    ssd_scan_prefill_fp32<H, P, N>(x_buf.data_, dt.data_, d_alog.data_, B_buf.data_, C_buf.data_,
                                   d_D.data_, d_dtb.data_, y_buf.data_, nullptr, B, S, G, 0.f, FLT_MAX);
    cudaDeviceSynchronize(); check("scan_y", y_buf.data_, scany_h);

    rmsnorm_gated_fp32<G>(y_buf.data_, gnormed.data_, d_gnw.data_, gate.data_, M, INTER, GROUP_SIZE, 1e-5f);
    cudaDeviceSynchronize(); check("gnormed", gnormed.data_, gnormed_h);

    gemm_fp32(gnormed.data_, d_opw.data_, mixer.data_, M, HIDDEN, INTER, true);
    cudaDeviceSynchronize(); check("mixer_out", mixer.data_, mixer_h);
}

// ===========================================================================
// 性能：全尺寸真实 config 单层 prefill 前向，扫描序列长度 S
//   维度取自 config.json（HIDDEN=3136 H=96 P=80 N=128 G=8，与真实模型一致）。
//   max_position_embeddings=262144（256K）在 8GB 4060 上单层微基准跑不了
//   （仅 in_proj 输出一个 [262144,17504] bf16 buffer 就 ~9.2GB），故扫
//   S∈{512,2048,8192} 看 scaling 与 fp8 何时反超 bf16（GEMM 转 compute-bound）。
//
// 计时口径：cudaEvent 仅包住 forward。workspace 复用：分配权重后 mark()，每次
//   forward 后 reset_to(mark) 回退偏移但保留 slab——warmup 首次 forward 生长
//   slab 池，之后重放相同分配序列命中已有 slab，计时区零 cudaMalloc/cudaFree。
//   显存因此只驻留「权重 + 单次 forward workspace」，大 S 也不膨胀。
// ===========================================================================
template<int HIDDEN, int H, int P, int N, int G>
static double bench_block_fp32(int S, int warm, int bench) {
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N,
                  PROJ = INTER + CONV_DIM + H, CONV_K = 4;
    const int B = 1, M = B * S;
    default_allocator().reset();
    auto z = [](int64_t n) { return allocate_tensor_zeros<float>(TensorShape::make_1d(n)); };
    auto bnw = z(HIDDEN); auto inw = z((int64_t)PROJ * HIDDEN);
    auto cw = z((int64_t)CONV_DIM * CONV_K); auto cb = z(CONV_DIM);
    auto alog = z(H); auto Dp = z(H); auto dtb = z(H); auto gnw = z(INTER);
    auto opw = z((int64_t)HIDDEN * INTER);
    auto d_in = z((int64_t)M * HIDDEN); auto d_out = z((int64_t)M * HIDDEN);
    cudaDeviceSynchronize();
    Mamba2BlockWeightsFP32 w{bnw.data_, inw.data_, cw.data_, cb.data_,
        alog.data_, Dp.data_, dtb.data_, gnw.data_, opw.data_};
    auto mk = default_allocator().mark();
    auto run = [&] { mamba_block_forward_fp32<HIDDEN, H, P, N, G>(d_in.data_, w, d_out.data_, B, S); };
    for (int i = 0; i < warm; ++i) { run(); default_allocator().reset_to(mk); }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float tot = 0.f;
    for (int i = 0; i < bench; ++i) {
        cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms;
        default_allocator().reset_to(mk);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tot / bench;
}

template<int HIDDEN, int H, int P, int N, int G>
static double bench_block_bf16(int S, int warm, int bench) {
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N,
                  PROJ = INTER + CONV_DIM + H, CONV_K = 4;
    using bf16 = __nv_bfloat16;
    const int B = 1, M = B * S;
    default_allocator().reset();
    auto zf = [](int64_t n) { return allocate_tensor_zeros<float>(TensorShape::make_1d(n)); };
    auto zb = [](int64_t n) { return allocate_tensor_zeros<bf16>(TensorShape::make_1d(n)); };
    auto bnw = zf(HIDDEN); auto cw = zf((int64_t)CONV_DIM * CONV_K); auto cb = zf(CONV_DIM);
    auto alog = zf(H); auto Dp = zf(H); auto dtb = zf(H); auto gnw = zf(INTER);
    auto inw = zb((int64_t)PROJ * HIDDEN); auto opw = zb((int64_t)HIDDEN * INTER);
    auto d_in = zb((int64_t)M * HIDDEN); auto d_out = zb((int64_t)M * HIDDEN);
    cudaDeviceSynchronize();
    Mamba2BlockWeightsBF16 w{bnw.data_, inw.data_, cw.data_, cb.data_,
        alog.data_, Dp.data_, dtb.data_, gnw.data_, opw.data_};
    auto mk = default_allocator().mark();
    auto run = [&] { mamba_block_forward_bf16<HIDDEN, H, P, N, G>(d_in.data_, w, d_out.data_, B, S); };
    for (int i = 0; i < warm; ++i) { run(); default_allocator().reset_to(mk); }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float tot = 0.f;
    for (int i = 0; i < bench; ++i) {
        cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms;
        default_allocator().reset_to(mk);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tot / bench;
}

template<int HIDDEN, int H, int P, int N, int G>
static double bench_block_fp8(int S, int warm, int bench) {
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N,
                  PROJ = INTER + CONV_DIM + H, CONV_K = 4;
    using bf16 = __nv_bfloat16; using fp8 = __nv_fp8_e4m3;
    const int B = 1, M = B * S;
    default_allocator().reset();
    auto zf = [](int64_t n) { return allocate_tensor_zeros<float>(TensorShape::make_1d(n)); };
    auto zb = [](int64_t n) { return allocate_tensor_zeros<bf16>(TensorShape::make_1d(n)); };
    auto z8 = [](int64_t n) { return allocate_tensor_zeros<fp8>(TensorShape::make_1d(n)); };
    auto bnw = zf(HIDDEN); auto cw = zf((int64_t)CONV_DIM * CONV_K); auto cb = zf(CONV_DIM);
    auto alog = zf(H); auto Dp = zf(H); auto dtb = zf(H); auto gnw = zf(INTER);
    auto inw = z8((int64_t)PROJ * HIDDEN); auto inws = zf(PROJ);
    auto opw = z8((int64_t)HIDDEN * INTER); auto opws = zf(HIDDEN);
    auto d_in = zb((int64_t)M * HIDDEN); auto d_out = zb((int64_t)M * HIDDEN);
    cudaDeviceSynchronize();
    Mamba2BlockWeightsFP8 w{bnw.data_, inw.data_, inws.data_, cw.data_, cb.data_,
        alog.data_, Dp.data_, dtb.data_, gnw.data_, opw.data_, opws.data_};
    auto mk = default_allocator().mark();
    auto run = [&] { mamba_block_forward_fp8<HIDDEN, H, P, N, G>(d_in.data_, w, d_out.data_, B, S); };
    for (int i = 0; i < warm; ++i) { run(); default_allocator().reset_to(mk); }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float tot = 0.f;
    for (int i = 0; i < bench; ++i) {
        cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms;
        default_allocator().reset_to(mk);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tot / bench;
}

TEST_F(MambaBlockTest, Perf_Sweep) {
    constexpr int HIDDEN = 3136, H = 96, P = 80, N = 128, G = 8;
    const int warm = 2, bench = 6;
    const int Ss[] = {512, 2048, 8192};
    printf("  [Perf sweep] HIDDEN=%d H=%d P=%d N=%d G=%d  B=1  (warm=%d bench=%d)\n",
           HIDDEN, H, P, N, G, warm, bench);
    printf("  %8s | %10s | %10s | %10s |  bf16x / fp8x\n",
           "S", "fp32(ms)", "bf16(ms)", "fp8(ms)");
    for (int S : Ss) {
        double t32 = bench_block_fp32<HIDDEN, H, P, N, G>(S, warm, bench);
        double t16 = bench_block_bf16<HIDDEN, H, P, N, G>(S, warm, bench);
        double t8  = bench_block_fp8 <HIDDEN, H, P, N, G>(S, warm, bench);
        printf("  %8d | %10.3f | %10.3f | %10.3f |  %.2fx / %.2fx\n",
               S, t32, t16, t8, t32 / t16, t32 / t8);
        EXPECT_GT(t16, 0.0);
    }
    default_allocator().reset();
}

// decode 单步延迟（全尺寸，B=1）：每 token 的 mamba 单层开销，bf16 vs fp8 权重对比。
// Mamba decode 是 O(1) 状态更新，与上下文长度无关——此延迟即稳态每 token 成本。
// decode 带宽受限：fp8 权重(1B) vs bf16(2B) 每 token 少读一半权重 → 预期 ~2x。
template<int HIDDEN, int H, int P, int N, int G>
static double bench_decode_bf16(int warm, int bench) {
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N, PROJ = INTER + CONV_DIM + H, CONV_K = 4;
    using bf16 = __nv_bfloat16;
    const int B = 1;
    default_allocator().reset();
    auto zf = [](int64_t n) { return allocate_tensor_zeros<float>(TensorShape::make_1d(n)); };
    auto zb = [](int64_t n) { return allocate_tensor_zeros<bf16>(TensorShape::make_1d(n)); };
    auto bnw = zf(HIDDEN); auto cw = zf((int64_t)CONV_DIM * CONV_K); auto cb = zf(CONV_DIM);
    auto alog = zf(H); auto Dp = zf(H); auto dtb = zf(H); auto gnw = zf(INTER);
    auto inw = zb((int64_t)PROJ * HIDDEN); auto opw = zb((int64_t)HIDDEN * INTER);
    auto conv_state = zf((int64_t)B * CONV_DIM * (CONV_K - 1)); auto ssm_state = zf((int64_t)B * H * P * N);
    auto d_in = zb((int64_t)B * HIDDEN); auto d_out = zb((int64_t)B * HIDDEN);
    cudaDeviceSynchronize();
    Mamba2BlockWeightsBF16 w{bnw.data_, inw.data_, cw.data_, cb.data_,
        alog.data_, Dp.data_, dtb.data_, gnw.data_, opw.data_};
    auto mk = default_allocator().mark();
    auto run = [&] { mamba_block_decode_bf16<HIDDEN, H, P, N, G>(d_in.data_, w, conv_state.data_, ssm_state.data_, d_out.data_, B); };
    for (int i = 0; i < warm; ++i) { run(); default_allocator().reset_to(mk); }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float tot = 0.f;
    for (int i = 0; i < bench; ++i) {
        cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms; default_allocator().reset_to(mk);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tot / bench;
}

template<int HIDDEN, int H, int P, int N, int G>
static double bench_decode_fp8(int warm, int bench) {
    constexpr int INTER = H * P, CONV_DIM = INTER + 2 * G * N, PROJ = INTER + CONV_DIM + H, CONV_K = 4;
    using bf16 = __nv_bfloat16; using fp8 = __nv_fp8_e4m3;
    const int B = 1;
    default_allocator().reset();
    auto zf = [](int64_t n) { return allocate_tensor_zeros<float>(TensorShape::make_1d(n)); };
    auto zb = [](int64_t n) { return allocate_tensor_zeros<bf16>(TensorShape::make_1d(n)); };
    auto z8 = [](int64_t n) { return allocate_tensor_zeros<fp8>(TensorShape::make_1d(n)); };
    auto bnw = zf(HIDDEN); auto cw = zf((int64_t)CONV_DIM * CONV_K); auto cb = zf(CONV_DIM);
    auto alog = zf(H); auto Dp = zf(H); auto dtb = zf(H); auto gnw = zf(INTER);
    auto inw = z8((int64_t)PROJ * HIDDEN); auto inws = zf(PROJ);
    auto opw = z8((int64_t)HIDDEN * INTER); auto opws = zf(HIDDEN);
    auto conv_state = zf((int64_t)B * CONV_DIM * (CONV_K - 1)); auto ssm_state = zf((int64_t)B * H * P * N);
    auto d_in = zb((int64_t)B * HIDDEN); auto d_out = zb((int64_t)B * HIDDEN);
    cudaDeviceSynchronize();
    Mamba2BlockWeightsFP8 w{bnw.data_, inw.data_, inws.data_, cw.data_, cb.data_,
        alog.data_, Dp.data_, dtb.data_, gnw.data_, opw.data_, opws.data_};
    auto mk = default_allocator().mark();
    auto run = [&] { mamba_block_decode_fp8<HIDDEN, H, P, N, G>(d_in.data_, w, conv_state.data_, ssm_state.data_, d_out.data_, B); };
    for (int i = 0; i < warm; ++i) { run(); default_allocator().reset_to(mk); }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float tot = 0.f;
    for (int i = 0; i < bench; ++i) {
        cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms; default_allocator().reset_to(mk);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tot / bench;
}

TEST_F(MambaBlockTest, Perf_Decode) {
    constexpr int HIDDEN = 3136, H = 96, P = 80, N = 128, G = 8;
    const int warm = 5, bench = 50;
    double t16 = bench_decode_bf16<HIDDEN, H, P, N, G>(warm, bench);
    double t8  = bench_decode_fp8 <HIDDEN, H, P, N, G>(warm, bench);
    printf("  [Perf] Mamba decode 1 token/layer:  bf16 %.4f ms  |  fp8 %.4f ms  (%.2fx)\n",
           t16, t8, t16 / t8);
    printf("         → 21 mamba 层/token:  bf16 %.2f ms  |  fp8 %.2f ms\n", t16 * 21, t8 * 21);
    EXPECT_GT(t16, 0.0); EXPECT_GT(t8, 0.0);
    default_allocator().reset();
}

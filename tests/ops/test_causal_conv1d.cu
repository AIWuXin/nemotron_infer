// ===========================================================================
// test_causal_conv1d.cu — Mamba-2 深度因果 Conv1D + SiLU
// ===========================================================================
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/mamba2/causal_conv1d.cuh"

using namespace nemotron;
using namespace nemotron::ops::mamba2;

namespace ref {

void conv1d_prefill_fp32(const float* x, const float* w, const float* b,
                         float* y, int S, int C) {
    for (int c = 0; c < C; ++c) {
        float st[3] = {0,0,0};
        for (int t = 0; t < S; ++t) {
            float x_t = x[t * C + c];
            float acc = b[c] + w[c*4+0]*st[0] + w[c*4+1]*st[1] + w[c*4+2]*st[2] + w[c*4+3]*x_t;
            float silu = acc / (1.f + std::exp(-acc));
            y[t * C + c] = silu;
            st[0] = st[1]; st[1] = st[2]; st[2] = x_t;
        }
    }
}

void conv1d_decode_fp32(float x, const float* w, const float* b,
                        float& y, float* state, int C, int c) {
    float acc = b[c] + w[c*4+0]*state[0] + w[c*4+1]*state[1] + w[c*4+2]*state[2] + w[c*4+3]*x;
    y = acc / (1.f + std::exp(-acc));
    state[0] = state[1]; state[1] = state[2]; state[2] = x;
}

}  // namespace ref

static void warmup_gpu() {
    float* buf; cudaMalloc(&buf, 1024); cudaMemset(buf, 0, 1024); cudaDeviceSynchronize(); cudaFree(buf);
}

class Conv1DTest : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); warmup_gpu(); }
};

TEST_F(Conv1DTest, FP32_Small) {
    const int C = 16, S = 8, B = 1;
    std::vector<float> x(B*S*C), w(C*4), b(C), y(B*S*C), exp(B*S*C);
    for (auto& v : x) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : w) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : b) v = float(rand())/RAND_MAX*0.1f;
    ref::conv1d_prefill_fp32(x.data(), w.data(), b.data(), exp.data(), S, C);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*S*C));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(C*4));
    auto d_b = allocate_tensor<float>(TensorShape::make_1d(C));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*C));
    copy_host_to_device(d_x, x.data()); copy_host_to_device(d_w, w.data());
    copy_host_to_device(d_b, b.data()); cudaDeviceSynchronize();

    causal_conv1d_prefill_fp32<16>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();

    copy_device_to_host(y.data(), d_y); cudaDeviceSynchronize();
    for (int i = 0; i < B*S*C; ++i)
        EXPECT_NEAR(y[i], exp[i], 1e-5f) << " i=" << i;

    free_tensor(d_x); free_tensor(d_w); free_tensor(d_b); free_tensor(d_y);
}

TEST_F(Conv1DTest, FP32_FullMamba) {
    const int C = 9728, S = 256, B = 1;
    std::vector<float> x(B*S*C), w(C*4), b(C), y(B*S*C), exp(B*S*C);
    for (auto& v : x) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : w) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : b) v = float(rand())/RAND_MAX*0.1f;
    ref::conv1d_prefill_fp32(x.data(), w.data(), b.data(), exp.data(), S, C);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*S*C));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(C*4));
    auto d_b = allocate_tensor<float>(TensorShape::make_1d(C));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*C));
    copy_host_to_device(d_x, x.data()); copy_host_to_device(d_w, w.data());
    copy_host_to_device(d_b, b.data()); cudaDeviceSynchronize();

    causal_conv1d_prefill_fp32<64>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();

    copy_device_to_host(y.data(), d_y); cudaDeviceSynchronize();
    for (int i = 0; i < B*S*C; ++i)
        EXPECT_NEAR(y[i], exp[i], 1e-5f) << " i=" << i;

    free_tensor(d_x); free_tensor(d_w); free_tensor(d_b); free_tensor(d_y);
}

TEST_F(Conv1DTest, BF16_Small) {
    // 覆盖向量化路径：conv_dim 偶数(走 bf162)、跨 chunk(S>TIME_CHUNK)、
    // 尾部 pair 越界 return(C/2=33 不整除 CH_TILE=64)
    const int C = 66, S = 130, B = 1;
    std::vector<float> x(B*S*C), w(C*4), b(C);
    for (auto& v : x) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : w) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : b) v = float(rand())/RAND_MAX*0.1f;

    // ref 用 bf16-round 后的 x，隔离输入量化误差，只比输出量化
    std::vector<float> xq(B*S*C);
    std::vector<__nv_bfloat16> x_bf(B*S*C);
    for (int i = 0; i < B*S*C; ++i) {
        x_bf[i] = __float2bfloat16_rn(x[i]);
        xq[i] = __bfloat162float(x_bf[i]);
    }
    std::vector<float> exp(B*S*C);
    ref::conv1d_prefill_fp32(xq.data(), w.data(), b.data(), exp.data(), S, C);

    auto d_x = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*S*C));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(C*4));
    auto d_b = allocate_tensor<float>(TensorShape::make_1d(C));
    auto d_y = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(B*S*C));
    copy_host_to_device(d_x, x_bf.data()); copy_host_to_device(d_w, w.data());
    copy_host_to_device(d_b, b.data()); cudaDeviceSynchronize();

    causal_conv1d_prefill_bf16<64>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> y_bf(B*S*C);
    copy_device_to_host(y_bf.data(), d_y); cudaDeviceSynchronize();
    for (int i = 0; i < B*S*C; ++i)
        EXPECT_NEAR(__bfloat162float(y_bf[i]), exp[i], 2e-2f) << " i=" << i;

    free_tensor(d_x); free_tensor(d_w); free_tensor(d_b); free_tensor(d_y);
}

TEST_F(Conv1DTest, Decode) {
    const int C = 16, B = 1;
    std::vector<float> w(C*4), b(C);
    for (auto& v : w) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : b) v = float(rand())/RAND_MAX*0.1f;

    float state_h[3*C] = {0};
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(C*4));
    auto d_b = allocate_tensor<float>(TensorShape::make_1d(C));
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(C*3));
    copy_host_to_device(d_w, w.data()); copy_host_to_device(d_b, b.data());
    copy_host_to_device(d_state, state_h);
    cudaDeviceSynchronize();

    // CPU 侧 state 与 GPU 同步演进（布局同 GPU：channel c 的 3 个 state 在 [c*3..c*3+2]）
    std::vector<float> cpu_state(C * 3, 0.f);
    for (int step = 0; step < 10; ++step) {
        float x_val = float(step) * 0.1f;
        auto x = std::vector<float>(B*C, x_val);
        auto y_gpu = std::vector<float>(B*C);
        auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*C));
        auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*C));
        copy_host_to_device(d_x, x.data()); cudaDeviceSynchronize();

        causal_conv1d_decode_fp32(d_x.data_, d_w.data_, d_b.data_, d_y.data_, d_state.data_, C, B);
        cudaDeviceSynchronize();

        copy_device_to_host(y_gpu.data(), d_y); cudaDeviceSynchronize();

        for (int c = 0; c < C; ++c) {
            float y_ref;
            ref::conv1d_decode_fp32(x_val, w.data(), b.data(), y_ref, &cpu_state[c*3], C, c);
            EXPECT_NEAR(y_gpu[c], y_ref, 1e-5f) << " step=" << step << " c=" << c;
        }
        free_tensor(d_x); free_tensor(d_y);
    }
    free_tensor(d_w); free_tensor(d_b); free_tensor(d_state);
}

// ===========================================================================
// 性能
// ===========================================================================
class Conv1DPerfTest : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); warmup_gpu(); }
    static constexpr int C = 9728, S = 8192, B = 1;
};

TEST_F(Conv1DPerfTest, FP32_Prefill) {
    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*S*C));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(C*4));
    auto d_b = allocate_tensor<float>(TensorShape::make_1d(C));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*C));

    int warm = 3, bench = 10;
    for (int i = 0; i < warm; ++i)
        causal_conv1d_prefill_fp32<64>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        causal_conv1d_prefill_fp32<64>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/bench;
    printf("  [Perf] Conv1D FP32 prefill  %6.3f ms  %d x %d\n", ms, C, S);
    free_tensor(d_x); free_tensor(d_w); free_tensor(d_b); free_tensor(d_y);
}

TEST_F(Conv1DPerfTest, BF16_Prefill) {
    auto d_x = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*S*C));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(C*4));
    auto d_b = allocate_tensor<float>(TensorShape::make_1d(C));
    auto d_y = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(B*S*C));

    int warm = 3, bench = 10;
    for (int i = 0; i < warm; ++i)
        causal_conv1d_prefill_bf16<64>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        causal_conv1d_prefill_bf16<64>(d_x.data_, d_w.data_, d_b.data_, d_y.data_, nullptr, C, S, B);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/bench;
    printf("  [Perf] Conv1D BF16 prefill  %6.3f ms  %d x %d\n", ms, C, S);
    free_tensor(d_x); free_tensor(d_w); free_tensor(d_b); free_tensor(d_y);
}

// ===========================================================================
// test_gemm.cu — GEMM 正确性测试（cuBLAS 封装验证）
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/gemm.cuh"

using namespace nemotron;
using namespace nemotron::ops;

// ===========================================================================
// 1. CPU 参考实现
// ===========================================================================
namespace ref {

void gemm_fp32(const float* x, const float* W, float* y,
               int M, int N, int K, bool transpose_W = true) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                float w_val = transpose_W ? W[n * K + k] : W[k * N + n];
                sum += x[m * K + k] * w_val;
            }
            y[m * N + n] = sum;
        }
    }
}

void gemm_bf16(const float* x, const float* W, float* y,
               int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                float x_bf = __bfloat162float(__float2bfloat16_rn(x[m * K + k]));
                float w_bf = __bfloat162float(__float2bfloat16_rn(W[n * K + k]));
                sum += x_bf * w_bf;
            }
            y[m * N + n] = __bfloat162float(__float2bfloat16_rn(sum));
        }
    }
}

}  // namespace ref

// ===========================================================================
// 2. 辅助
// ===========================================================================
static void warmup_gpu() {
    float* buf = nullptr;
    cudaMalloc(&buf, 1024);
    cudaMemset(buf, 0, 1024);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

// ===========================================================================
// 3. FP32 GEMM 正确性
// ===========================================================================
class GemmFP32Test : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(GemmFP32Test, Tiny) {
    const int M = 2, N = 3, K = 4;

    std::vector<float> x = {1,2,3,4, 5,6,7,8};                 // [2,4]
    std::vector<float> W = {1,0,0,0, 0,1,0,0, 0,0,1,0};      // [3,4]
    std::vector<float> expected(M * N), out(M * N);

    ref::gemm_fp32(x.data(), W.data(), expected.data(), M, N, K, true);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(M * K));
    auto d_W = allocate_tensor<float>(TensorShape::make_1d(N * K));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(M * N));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_W, W.data());
    cudaDeviceSynchronize();

    gemm_fp32(d_x.data_, d_W.data_, d_y.data_, M, N, K);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i)
        EXPECT_FLOAT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_x); free_tensor(d_W); free_tensor(d_y);
}

TEST_F(GemmFP32Test, SmallMLP) {
    // 模拟 MLP 维度: B=2, S=64, hidden=128, inter=256 (缩小版)
    const int M = 128, N = 256, K = 128;

    std::vector<float> x(M * K), W(N * K), expected(M * N), out(M * N);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(rand()) / RAND_MAX - 0.5f;
    for (size_t i = 0; i < W.size(); ++i) W[i] = float(rand()) / RAND_MAX - 0.5f;

    ref::gemm_fp32(x.data(), W.data(), expected.data(), M, N, K, true);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(M * K));
    auto d_W = allocate_tensor<float>(TensorShape::make_1d(N * K));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(M * N));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_W, W.data());
    cudaDeviceSynchronize();

    gemm_fp32(d_x.data_, d_W.data_, d_y.data_, M, N, K);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-3f) << " at index " << i;

    free_tensor(d_x); free_tensor(d_W); free_tensor(d_y);
}

// ===========================================================================
// 4. BF16 GEMM 正确性
// ===========================================================================
class GemmBF16Test : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(GemmBF16Test, Small) {
    const int M = 64, N = 128, K = 256;

    std::vector<float> x_fp32(M * K), W_fp32(N * K), expected(M * N);
    for (size_t i = 0; i < x_fp32.size(); ++i) x_fp32[i] = float(rand()) / RAND_MAX - 0.5f;
    for (size_t i = 0; i < W_fp32.size(); ++i) W_fp32[i] = float(rand()) / RAND_MAX - 0.5f;

    ref::gemm_bf16(x_fp32.data(), W_fp32.data(), expected.data(), M, N, K);

    std::vector<bfloat16_t> x_bf16(M * K), W_bf16(N * K);
    for (size_t i = 0; i < x_bf16.size(); ++i) x_bf16[i] = __float2bfloat16_rn(x_fp32[i]);
    for (size_t i = 0; i < W_bf16.size(); ++i) W_bf16[i] = __float2bfloat16_rn(W_fp32[i]);

    auto d_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(M * K));
    auto d_W = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N * K));
    auto d_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(M * N));
    copy_host_to_device(d_x, x_bf16.data());
    copy_host_to_device(d_W, W_bf16.data());
    cudaDeviceSynchronize();

    gemm_bf16(d_x.data_, d_W.data_, d_y.data_, M, N, K);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(M * N);
    copy_device_to_host(out_bf16.data(), d_y);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out_f, expected[i], 0.02f) << " at index " << i;
    }

    free_tensor(d_x); free_tensor(d_W); free_tensor(d_y);
}

// ===========================================================================
// 5. 性能基准 + 对比
// ===========================================================================
class GemmComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!fp32_x.data_) {
            fp32_x = allocate_tensor<float>(TensorShape::make_1d(M * K));
            fp32_W = allocate_tensor<float>(TensorShape::make_1d(N * K));
            fp32_y = allocate_tensor_zeros<float>(TensorShape::make_1d(M * N));
        }
        if (!bf16_x.data_) {
            bf16_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(M * K));
            bf16_W = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N * K));
            bf16_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(M * N));
        }

        // FP32 x data
        std::vector<float> h_x(M * K);
        for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = float(i) * 0.001f - 3.f;
        copy_host_to_device(fp32_x, h_x.data());

        // FP32 W data (N*K 可能很大，不预存到 vector，直接用小的 seed 填充)
        std::vector<float> h_W(N * K);
        for (size_t i = 0; i < h_W.size(); ++i) h_W[i] = float(i) * 0.002f + 1.f;
        copy_host_to_device(fp32_W, h_W.data());

        // BF16 x + W data
        std::vector<bfloat16_t> h_bf16_x(M * K);
        for (size_t i = 0; i < h_bf16_x.size(); ++i) h_bf16_x[i] = __float2bfloat16_rn(h_x[i]);
        copy_host_to_device(bf16_x, h_bf16_x.data());

        std::vector<bfloat16_t> h_bf16_W(N * K);
        for (size_t i = 0; i < h_bf16_W.size(); ++i) h_bf16_W[i] = __float2bfloat16_rn(h_W[i]);
        copy_host_to_device(bf16_W, h_bf16_W.data());

        cudaDeviceSynchronize();
    }

    // 真实 MLP up_proj 维度: B=2, S=128, hidden=3136, inter=12544
    static constexpr int M = 256;       // B*S
    static constexpr int N = 12544;     // output_dim
    static constexpr int K = 3136;      // hidden_size

    Tensor<float> fp32_x, fp32_W, fp32_y;
    Tensor<bfloat16_t> bf16_x, bf16_W, bf16_y;
};

TEST_F(GemmComparisonTest, MLP_UpProj_Speedup) {
    int warmup_iters = 3, bench_iters = 10;

    // FP32
    for (int i = 0; i < warmup_iters; ++i)
        gemm_fp32(fp32_x.data_, fp32_W.data_, fp32_y.data_, M, N, K);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        gemm_fp32(fp32_x.data_, fp32_W.data_, fp32_y.data_, M, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    // BF16
    for (int i = 0; i < warmup_iters; ++i)
        gemm_bf16(bf16_x.data_, bf16_W.data_, bf16_y.data_, M, N, K);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        gemm_bf16(bf16_x.data_, bf16_W.data_, bf16_y.data_, M, N, K);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    double speedup = fp32_ms / bf16_ms;
    printf("  [Compare] GEMM MLP up_proj   FP32: %6.3f ms | BF16: %6.3f ms | Speedup: %.2f x\n",
           fp32_ms, bf16_ms, speedup);
    EXPECT_GT(speedup, 0.5);
}

// ===========================================================================
// 6. FP8 GEMM 正确性
// ===========================================================================

namespace ref {

// CPU 参考: y = (x_fp8 * x_scale) @ (W_fp8 * w_scale)^T
//   w_scale 沿输出通道 N（per-row），与算子的 epilogue 列 scale 对齐
void gemm_fp8_ref(const __nv_fp8_e4m3* x_fp8, float x_scale,
                  const __nv_fp8_e4m3* W_fp8, const float* w_scale,
                  float* y, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                float x_val = static_cast<float>(x_fp8[m * K + k]) * x_scale;
                float w_val = static_cast<float>(W_fp8[n * K + k]);
                sum += x_val * w_val;
            }
            y[m * N + n] = sum * w_scale[n];
        }
    }
}

// CPU 侧量化: FP32 → FP8
float quant_fp32_to_fp8(const float* src, __nv_fp8_e4m3* dst, int N) {
    float amax = 0.f;
    for (int i = 0; i < N; ++i) amax = std::max(amax, std::fabs(src[i]));
    float scale = amax / 448.f;
    if (scale < 1e-10f) scale = 1.f / 448.f;
    float inv_scale = 1.f / scale;
    for (int i = 0; i < N; ++i) {
        float q = src[i] * inv_scale;
        q = std::max(-448.f, std::min(448.f, q));
        dst[i] = static_cast<__nv_fp8_e4m3>(q);
    }
    return scale;
}

}  // namespace ref

class GemmFP8Test : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(GemmFP8Test, Tiny) {
    const int M = 4, N = 8, K = 16;

    // 生成 FP32 数据
    std::vector<float> x_fp32(M * K), W_fp32(N * K);
    for (size_t i = 0; i < x_fp32.size(); ++i) x_fp32[i] = float(i % 7) * 0.3f - 1.f;
    for (size_t i = 0; i < W_fp32.size(); ++i) W_fp32[i] = float(i % 11) * 0.2f - 1.1f;

    // CPU 量化
    std::vector<__nv_fp8_e4m3> x_fp8(M * K), W_fp8_raw(N * K);
    float x_scale = ref::quant_fp32_to_fp8(x_fp32.data(), x_fp8.data(), M * K);

    // 权重预处理: per-row(输出通道 N) scale
    std::vector<__nv_fp8_e4m3> W_fp8(N * K);
    std::vector<float> w_scale(N);
    for (int n = 0; n < N; ++n) {
        float row_max = 0.f;
        for (int k = 0; k < K; ++k)
            row_max = std::max(row_max, std::fabs(W_fp32[n * K + k]));
        w_scale[n] = row_max / 448.f;
        if (w_scale[n] < 1e-10f) w_scale[n] = 1.f / 448.f;
        float inv = 1.f / w_scale[n];
        for (int k = 0; k < K; ++k) {
            float q = W_fp32[n * K + k] * inv;
            q = std::max(-448.f, std::min(448.f, q));
            W_fp8[n * K + k] = static_cast<__nv_fp8_e4m3>(q);
        }
    }

    // CPU 参考
    std::vector<float> expected(M * N);
    ref::gemm_fp8_ref(x_fp8.data(), x_scale, W_fp8.data(), w_scale.data(),
                      expected.data(), M, N, K);

    // GPU
    auto d_x  = allocate_tensor<__nv_fp8_e4m3>(TensorShape::make_1d(M * K));
    auto d_W  = allocate_tensor<__nv_fp8_e4m3>(TensorShape::make_1d(N * K));
    auto d_y  = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(M * N));
    auto d_ws = allocate_tensor<float>(TensorShape::make_1d(N));   // per-row weight scale
    auto d_as = allocate_tensor<float>(TensorShape::make_1d(1));  // device x_scale
    const size_t ws_bytes = 4 * 1024 * 1024;
    auto d_wsbuf = allocate_tensor<char>(TensorShape::make_1d(ws_bytes));  // cuBLASLt workspace
    copy_host_to_device(d_x, x_fp8.data());
    copy_host_to_device(d_W, W_fp8.data());
    copy_host_to_device(d_ws, w_scale.data());
    copy_host_to_device(d_as, &x_scale);
    cudaDeviceSynchronize();

    gemm_fp8(d_x.data_, d_W.data_, d_y.data_, d_as.data_, d_ws.data_, M, N, K,
             d_wsbuf.data_, ws_bytes);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out_bf16(M * N);
    copy_device_to_host(out_bf16.data(), d_y);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        // FP8 精度有限，且 cuBLASLt 内部量化路径与 CPU 参考不同
        EXPECT_NEAR(out_f, expected[i], 0.15f) << " at index " << i;
    }

    free_tensor(d_x); free_tensor(d_W); free_tensor(d_y); free_tensor(d_ws); free_tensor(d_as);
}

// ===========================================================================
// 7. FP32 vs BF16 vs FP8 三路对比
// ===========================================================================
class GemmThreeWayTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!fp32_x.data_) {
            fp32_x = allocate_tensor<float>(TensorShape::make_1d(M * K));
            fp32_W = allocate_tensor<float>(TensorShape::make_1d(N * K));
            fp32_y = allocate_tensor_zeros<float>(TensorShape::make_1d(M * N));
        }
        if (!bf16_x.data_) {
            bf16_x = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(M * K));
            bf16_W = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N * K));
            bf16_y = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(M * N));
        }
        if (!fp8_x.data_) {
            fp8_x  = allocate_tensor<__nv_fp8_e4m3>(TensorShape::make_1d(M * K));
            fp8_W  = allocate_tensor<__nv_fp8_e4m3>(TensorShape::make_1d(N * K));
            fp8_y  = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(M * N));
            fp8_ws = allocate_tensor<float>(TensorShape::make_1d(N));
            fp8_as = allocate_tensor<float>(TensorShape::make_1d(1));
            fp8_wsbuf = allocate_tensor<char>(TensorShape::make_1d(kWsBytes));
        }

        // 生成数据
        std::vector<float> h_x(M * K), h_W(N * K);
        for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = float(i) * 0.001f - 3.f;
        for (size_t i = 0; i < h_W.size(); ++i) h_W[i] = float(i) * 0.002f + 1.f;
        copy_host_to_device(fp32_x, h_x.data());
        copy_host_to_device(fp32_W, h_W.data());

        // BF16
        std::vector<__nv_bfloat16> h_bf16(M * K), hW_bf16(N * K);
        for (size_t i = 0; i < h_bf16.size(); ++i) h_bf16[i] = __float2bfloat16_rn(h_x[i]);
        for (size_t i = 0; i < hW_bf16.size(); ++i) hW_bf16[i] = __float2bfloat16_rn(h_W[i]);
        copy_host_to_device(bf16_x, h_bf16.data());
        copy_host_to_device(bf16_W, hW_bf16.data());

        // FP8 (CPU 侧量化: activation per-tensor, weight per-column)
        std::vector<__nv_fp8_e4m3> h_fp8_x(M * K);
        float x_scale = ref::quant_fp32_to_fp8(h_x.data(), h_fp8_x.data(), M * K);
        copy_host_to_device(fp8_x, h_fp8_x.data());
        copy_host_to_device(fp8_as, &x_scale);

        std::vector<__nv_fp8_e4m3> h_fp8_W(N * K);
        std::vector<float> h_ws(N);
        for (int n = 0; n < N; ++n) {
            float row_max = 0.f;
            for (int k = 0; k < K; ++k)
                row_max = std::max(row_max, std::fabs(h_W[n * K + k]));
            h_ws[n] = row_max / 448.f;
            if (h_ws[n] < 1e-10f) h_ws[n] = 1.f / 448.f;
            float inv = 1.f / h_ws[n];
            for (int k = 0; k < K; ++k) {
                float q = h_W[n * K + k] * inv;
                q = std::max(-448.f, std::min(448.f, q));
                h_fp8_W[n * K + k] = static_cast<__nv_fp8_e4m3>(q);
            }
        }
        copy_host_to_device(fp8_W, h_fp8_W.data());
        copy_host_to_device(fp8_ws, h_ws.data());
        cudaDeviceSynchronize();
    }

    static constexpr int M = 256;
    static constexpr int N = 12544;
    static constexpr int K = 3136;

    Tensor<float> fp32_x, fp32_W, fp32_y;
    Tensor<__nv_bfloat16> bf16_x, bf16_W, bf16_y;
    Tensor<__nv_fp8_e4m3> fp8_x, fp8_W;
    Tensor<__nv_bfloat16> fp8_y;
    Tensor<float> fp8_ws, fp8_as;
    Tensor<char> fp8_wsbuf;                       // cuBLASLt workspace
    static constexpr size_t kWsBytes = 32 * 1024 * 1024;
};

TEST_F(GemmThreeWayTest, MLP_UpProj_Speedup) {
    int warmup_iters = 3, bench_iters = 10;

    // FP32
    for (int i = 0; i < warmup_iters; ++i)
        gemm_fp32(fp32_x.data_, fp32_W.data_, fp32_y.data_, M, N, K);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        gemm_fp32(fp32_x.data_, fp32_W.data_, fp32_y.data_, M, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    // BF16
    for (int i = 0; i < warmup_iters; ++i)
        gemm_bf16(bf16_x.data_, bf16_W.data_, bf16_y.data_, M, N, K);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        gemm_bf16(bf16_x.data_, bf16_W.data_, bf16_y.data_, M, N, K);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    // FP8
    for (int i = 0; i < warmup_iters; ++i)
        gemm_fp8(fp8_x.data_, fp8_W.data_, fp8_y.data_, fp8_as.data_, fp8_ws.data_, M, N, K,
                 fp8_wsbuf.data_, kWsBytes);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        gemm_fp8(fp8_x.data_, fp8_W.data_, fp8_y.data_, fp8_as.data_, fp8_ws.data_, M, N, K,
                 fp8_wsbuf.data_, kWsBytes);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double fp8_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    printf("  [Compare] GEMM MLP up_proj   FP32: %6.3f ms | BF16: %6.3f ms | FP8: %6.3f ms | Speedup(BF16): %.2f x | Speedup(FP8): %.2f x\n",
           fp32_ms, bf16_ms, fp8_ms, fp32_ms / bf16_ms, fp32_ms / fp8_ms);
    EXPECT_GT(fp8_ms, 0.0);
}

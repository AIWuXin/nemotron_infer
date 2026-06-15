// ===========================================================================
// test_sdpa_prefill.cu — SDPA FlashAttention 测试
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/attention/sdpa_prefill.cuh"

using namespace nemotron;
using namespace nemotron::ops::attention;

// ===========================================================================
// 1. CPU 参考
// ===========================================================================
namespace ref {

void sdpa_prefill_fp32(
    const float* Q, const float* K, const float* V, float* O,
    int S, int H, int head_dim, bool causal
) {
    const float rsqrt_d = 1.f / std::sqrt((float)head_dim);
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < S; ++t) {
            // score[t] = Q[t] @ K^T / sqrt(d)
            std::vector<float> score(S);
            for (int s = 0; s < S; ++s) {
                float sum = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    sum += Q[h * S * head_dim + t * head_dim + d]
                         * K[h * S * head_dim + s * head_dim + d];
                score[s] = sum * rsqrt_d;
                if (causal && s > t) score[s] = -INFINITY;
            }

            // softmax
            float max_val = -INFINITY;
            for (int s = 0; s < S; ++s) max_val = std::max(max_val, score[s]);
            float sum = 0.f;
            for (int s = 0; s < S; ++s) sum += std::exp(score[s] - max_val);
            for (int s = 0; s < S; ++s) score[s] = std::exp(score[s] - max_val) / sum;

            // output = Σ score[s] * V[s]
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.f;
                for (int s = 0; s < S; ++s)
                    val += score[s] * V[h * S * head_dim + s * head_dim + d];
                O[h * S * head_dim + t * head_dim + d] = val;
            }
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

static std::vector<float> rand_vec(size_t n, float range = 1.f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = (float(rand()) / RAND_MAX * 2.f - 1.f) * range;
    return v;
}

// ===========================================================================
// 3. FP32 正确性
// ===========================================================================
class SDPAPrefillFP32Test : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(SDPAPrefillFP32Test, SmallCausal) {
    const int S = 16, H = 2, head_dim = 128;
    const int N = H * S * head_dim;

    auto Q_h = rand_vec(N);
    auto K_h = rand_vec(N);
    auto V_h = rand_vec(N);
    std::vector<float> expected(N), out(N);

    ref::sdpa_prefill_fp32(Q_h.data(), K_h.data(), V_h.data(), expected.data(), S, H, head_dim, true);

    auto d_Q = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_K = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_V = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_O = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_Q, Q_h.data());
    copy_host_to_device(d_K, K_h.data());
    copy_host_to_device(d_V, V_h.data());
    cudaDeviceSynchronize();

    sdpa_prefill_fp32(d_Q.data_, d_K.data_, d_V.data_, d_O.data_, S, H);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_O);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-4f) << " at index " << i;

    free_tensor(d_Q); free_tensor(d_K); free_tensor(d_V); free_tensor(d_O);
}

TEST_F(SDPAPrefillFP32Test, FullCausal) {
    const int S = 128, H = 4, head_dim = 128;
    const int N = H * S * head_dim;

    auto Q_h = rand_vec(N, 0.5f);
    auto K_h = rand_vec(N, 0.5f);
    auto V_h = rand_vec(N, 0.5f);
    std::vector<float> expected(N), out(N);

    ref::sdpa_prefill_fp32(Q_h.data(), K_h.data(), V_h.data(), expected.data(), S, H, head_dim, true);

    auto d_Q = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_K = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_V = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_O = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_Q, Q_h.data());
    copy_host_to_device(d_K, K_h.data());
    copy_host_to_device(d_V, V_h.data());
    cudaDeviceSynchronize();

    sdpa_prefill_fp32(d_Q.data_, d_K.data_, d_V.data_, d_O.data_, S, H);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_O);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-4f) << " at index " << i;

    free_tensor(d_Q); free_tensor(d_K); free_tensor(d_V); free_tensor(d_O);
}

// ===========================================================================
// 4. BF16 正确性
// ===========================================================================
class SDPAPrefillBF16Test : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(SDPAPrefillBF16Test, SmallCausal) {
    const int S = 8, H = 2, head_dim = 128;
    const int N = H * S * head_dim;

    auto Q_fp32 = rand_vec(N, 0.5f);
    auto K_fp32 = rand_vec(N, 0.5f);
    auto V_fp32 = rand_vec(N, 0.5f);

    // CPU reference (FP32)
    std::vector<float> expected_fp32(N);
    ref::sdpa_prefill_fp32(Q_fp32.data(), K_fp32.data(), V_fp32.data(), expected_fp32.data(), S, H, head_dim, true);

    // BF16 GPU
    std::vector<__nv_bfloat16> Q_bf16(N), K_bf16(N), V_bf16(N);
    for (int i = 0; i < N; ++i) {
        Q_bf16[i] = __float2bfloat16_rn(Q_fp32[i]);
        K_bf16[i] = __float2bfloat16_rn(K_fp32[i]);
        V_bf16[i] = __float2bfloat16_rn(V_fp32[i]);
    }

    auto d_Q = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N));
    auto d_K = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N));
    auto d_V = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N));
    auto d_O = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(N));
    copy_host_to_device(d_Q, Q_bf16.data());
    copy_host_to_device(d_K, K_bf16.data());
    copy_host_to_device(d_V, V_bf16.data());
    cudaDeviceSynchronize();

    sdpa_prefill_bf16(d_Q.data_, d_K.data_, d_V.data_, d_O.data_, S, H);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_O);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(__bfloat162float(out_bf16[i]), expected_fp32[i], 0.1f) << " at index " << i;

    free_tensor(d_Q); free_tensor(d_K); free_tensor(d_V); free_tensor(d_O);
}

// ===========================================================================
// 5. 性能基准 + FP32 vs BF16 对比
// ===========================================================================
class SDPAPrefillComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!fp32_Q.data_) {
            fp32_Q = allocate_tensor<float>(TensorShape::make_1d(N));
            fp32_K = allocate_tensor<float>(TensorShape::make_1d(N));
            fp32_V = allocate_tensor<float>(TensorShape::make_1d(N));
            fp32_O = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
        }
        if (!bf16_Q.data_) {
            bf16_Q = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N));
            bf16_K = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N));
            bf16_V = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(N));
            bf16_O = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(N));
        }

        auto h = rand_vec(N, 0.5f);
        copy_host_to_device(fp32_Q, h.data());
        copy_host_to_device(fp32_K, h.data());
        copy_host_to_device(fp32_V, h.data());

        std::vector<__nv_bfloat16> h_bf16(N);
        for (int i = 0; i < N; ++i) h_bf16[i] = __float2bfloat16_rn(h[i]);
        copy_host_to_device(bf16_Q, h_bf16.data());
        copy_host_to_device(bf16_K, h_bf16.data());
        copy_host_to_device(bf16_V, h_bf16.data());
        cudaDeviceSynchronize();
    }

    // 模拟真实维度: H=4, S=256（全量保存时序）
    static constexpr int S = 256;
    static constexpr int H = 40;
    static constexpr int HEAD = 128;
    static constexpr int N = H * S * HEAD;

    Tensor<float> fp32_Q, fp32_K, fp32_V, fp32_O;
    Tensor<__nv_bfloat16> bf16_Q, bf16_K, bf16_V, bf16_O;
};

TEST_F(SDPAPrefillComparisonTest, Speedup) {
    int warmup_iters = 10, bench_iters = 50;

    // FP32
    for (int i = 0; i < warmup_iters; ++i)
        sdpa_prefill_fp32(fp32_Q.data_, fp32_K.data_, fp32_V.data_, fp32_O.data_, S, H);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        sdpa_prefill_fp32(fp32_Q.data_, fp32_K.data_, fp32_V.data_, fp32_O.data_, S, H);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    // BF16
    for (int i = 0; i < warmup_iters; ++i)
        sdpa_prefill_bf16(bf16_Q.data_, bf16_K.data_, bf16_V.data_, bf16_O.data_, S, H);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        sdpa_prefill_bf16(bf16_Q.data_, bf16_K.data_, bf16_V.data_, bf16_O.data_, S, H);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    double speedup = fp32_ms / bf16_ms;
    printf("  [Compare] SDPA Prefill      FP32: %6.3f ms | BF16: %6.3f ms | Speedup: %.2f x\n",
           fp32_ms, bf16_ms, speedup);
    EXPECT_GT(speedup, 0.5);
}

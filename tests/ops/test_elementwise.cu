// ===========================================================================
// test_elementwise.cu — Phase 2 逐元素算子测试（正确性 + 性能）
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/elementwise.cuh"

using namespace nemotron;
using namespace nemotron::ops;

// ===========================================================================
// 1. CPU 参考实现（用于正确性对比）
// ===========================================================================
namespace ref {

inline float softplus(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return 0.f;
    return std::log1p(std::exp(x));  // log1p(exp) 比 log(1+exp) 更精确
}

inline float silu(float x) {
    return x / (1.f + std::exp(-x));
}

inline float relu2(float x) {
    return x > 0.f ? x * x : 0.f;
}

inline float clamp_softplus(float x, float bias,
                             float clamp_min, float clamp_max) {
    float v = softplus(x + bias);
    return v < clamp_min ? clamp_min : (v > clamp_max ? clamp_max : v);
}

}  // namespace ref

// ===========================================================================
// 2. 测试辅助函数
// ===========================================================================

/// GPU 上预热运行（消除首次 kernel launch 开销）
static void warmup_gpu() {
    float *buf = nullptr;
    cudaMalloc(&buf, 1024);
    cudaMemset(buf, 0, 1024);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

/// 分配 + 拷贝 + 运行 + 拷贝回，返回耗时 (ms)
template<ElementwiseType Op>
double run_and_measure(float* d_out, float* d_in0, float* d_in1,
                       size_t size, float scale = 1.f,
                        float* d_bias = nullptr,
                        float clamp_min = 0.f, float clamp_max = 1.f,
                        int warmup_iters = 5, int bench_iters = 20) {
    cudaStream_t stream = 0;
    // 预热
    for (int i = 0; i < warmup_iters; ++i) {
        elementwise_ops_fp32<Op>(d_in0, d_in1, d_out, size,
                                  scale, d_bias, clamp_min, clamp_max, stream);
    }
    cudaDeviceSynchronize();

    // 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        elementwise_ops_fp32<Op>(d_in0, d_in1, d_out, size,
                                  scale, d_bias, clamp_min, clamp_max, stream);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
}

// ===========================================================================
// 3. 正确性测试 — 小数组全覆盖
// ===========================================================================

class ElementwiseCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override { warmup_gpu(); }

    // 用 1024 个元素覆盖 4 倍数 + 余数 3（测试向量化和尾数阶段）
    static constexpr size_t N = 1027;
};

// ---- Add ----
TEST_F(ElementwiseCorrectnessTest, Add) {
    std::vector<float> in0(N), in1(N), out(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        in0[i] = float(i) * 0.25f;
        in1[i] = float(i) * 0.1f - 10.f;
        expected[i] = in0[i] + in1[i];
    }
    auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_in1 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0.data());
    copy_host_to_device(d_in1, in1.data());

    elementwise_ops_fp32<kElementwiseAdd>(
        d_in0.data_, d_in1.data_, d_out.data_, N);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_in0); free_tensor(d_in1); free_tensor(d_out);
}

// ---- Mul ----
TEST_F(ElementwiseCorrectnessTest, Mul) {
    std::vector<float> in0(N), in1(N), out(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        in0[i] = float(i) * 0.1f - 50.f;
        in1[i] = float(i) * 0.05f + 2.f;
        expected[i] = in0[i] * in1[i];
    }
    auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_in1 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0.data());
    copy_host_to_device(d_in1, in1.data());

    elementwise_ops_fp32<kElementwiseMul>(
        d_in0.data_, d_in1.data_, d_out.data_, N);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_in0); free_tensor(d_in1); free_tensor(d_out);
}

// ---- Scale ----
TEST_F(ElementwiseCorrectnessTest, Scale) {
    const float alpha = 2.71828f;
    std::vector<float> in0(N), out(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        in0[i] = float(i) * 0.5f - 100.f;
        expected[i] = in0[i] * alpha;
    }
    auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0.data());

    elementwise_ops_fp32<kElementwiseScale>(
        d_in0.data_, nullptr, d_out.data_, N, alpha);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_in0); free_tensor(d_out);
}

// ---- ReLU² ----
TEST_F(ElementwiseCorrectnessTest, Relu2) {
    std::vector<float> in0(N), out(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        in0[i] = float(i) * 0.1f - 51.f;  // 一半负一半正
        expected[i] = ref::relu2(in0[i]);
    }
    auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0.data());

    elementwise_ops_fp32<kElementwiseRelu2>(
        d_in0.data_, nullptr, d_out.data_, N);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_in0); free_tensor(d_out);
}

// ---- SiLU ----
TEST_F(ElementwiseCorrectnessTest, Silu) {
    std::vector<float> in0(N), out(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        in0[i] = float(i) * 0.05f - 25.f;
        expected[i] = ref::silu(in0[i]);
    }
    auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0.data());

    elementwise_ops_fp32<kElementwiseSilu>(
        d_in0.data_, nullptr, d_out.data_, N);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    // GPU __expf() 精度略低于 CPU std::exp()（差异约 1-2 ULP~1e-10），
    // 不能用 EXPECT_FLOAT_EQ，需要给 1e-6f 容忍度
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-6f) << " at index " << i;

    free_tensor(d_in0); free_tensor(d_out);
}

// ---- ClampSoftplus ----
TEST_F(ElementwiseCorrectnessTest, ClampSoftplus) {
    const float c_min = 0.001f, c_max = 0.1f;
    std::vector<float> in0(N), bias(N), out(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        in0[i] = float(i) * 0.02f - 10.f;
        bias[i] = float(i % 96) * 0.01f - 0.48f;  // 模拟 96 个 dt_bias 值
        expected[i] = ref::clamp_softplus(in0[i], bias[i], c_min, c_max);
    }
    auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_bias = allocate_tensor<float>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0.data());
    copy_host_to_device(d_bias, bias.data());

    elementwise_ops_fp32<kElementwiseClampSoftplus>(
        d_in0.data_, nullptr, d_out.data_, N,
        1.f /*unused*/, d_bias.data_, c_min, c_max);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_in0); free_tensor(d_bias); free_tensor(d_out);
}

// ---- 尾数阶段专项测试（size 非 4 的倍数） ----
TEST_F(ElementwiseCorrectnessTest, RemainderElements) {
    // 用刚好的小数组测试：0~3 个余数
    for (size_t Nt : {1, 2, 3, 4, 5, 7, 15, 63, 255, 1021, 4095}) {
        std::vector<float> in0(Nt), out(Nt), expected(Nt);
        for (size_t i = 0; i < Nt; ++i) {
            in0[i] = float(i) * 0.1f - 5.f;
            expected[i] = ref::relu2(in0[i]);
        }
        auto d_in0 = allocate_tensor<float>(TensorShape::make_1d(Nt));
        auto d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(Nt));
        copy_host_to_device(d_in0, in0.data());

        elementwise_ops_fp32<kElementwiseRelu2>(
            d_in0.data_, nullptr, d_out.data_, Nt);
        cudaDeviceSynchronize();

        copy_device_to_host(out.data(), d_out);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < Nt; ++i)
            EXPECT_FLOAT_EQ(out[i], expected[i]) << " size=" << Nt << " index=" << i;

        free_tensor(d_in0); free_tensor(d_out);
    }
}

// ===========================================================================
// 4. 性能基准测试
// ===========================================================================

class ElementwisePerfTest : public ::testing::Test {
protected:
    void SetUp() override { warmup_gpu(); }

    static constexpr size_t PERF_N = 32ULL * 1024 * 1024;  // 32M elements = 128 MB
    static constexpr size_t BYTES  = PERF_N * sizeof(float);

    // 预分配，所有性能测试共用，减少分配噪音
    void init_buffers() {
        if (!d_in0.data_) {
            d_in0 = allocate_tensor<float>(TensorShape::make_1d(PERF_N));
        }
        if (!d_in1.data_) {
            d_in1 = allocate_tensor<float>(TensorShape::make_1d(PERF_N));
        }
        if (!d_out.data_) {
            d_out = allocate_tensor_zeros<float>(TensorShape::make_1d(PERF_N));
        }
        h_buf.resize(PERF_N);
        for (size_t i = 0; i < PERF_N; ++i) h_buf[i] = float(i) * 0.001f - 5.f;
        copy_host_to_device(d_in0, h_buf.data());
        copy_host_to_device(d_in1, h_buf.data());
        cudaDeviceSynchronize();
    }

    Tensor<float> d_in0, d_in1, d_out;
    std::vector<float> h_buf;
};

TEST_F(ElementwisePerfTest, Add_Bandwidth) {
    init_buffers();
    double ms = run_and_measure<kElementwiseAdd>(
        d_out.data_, d_in0.data_, d_in1.data_, PERF_N);
    // 读 2 个输入 + 写 1 个输出 = 3 * 128 MB = 384 MB
    double bw = (3.0 * BYTES) / (ms * 1e6);  // GB/s
    printf("  [Perf] Add:            %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 3.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 100.0);
}

TEST_F(ElementwisePerfTest, Mul_Bandwidth) {
    init_buffers();
    double ms = run_and_measure<kElementwiseMul>(
        d_out.data_, d_in0.data_, d_in1.data_, PERF_N);
    double bw = (3.0 * BYTES) / (ms * 1e6);
    printf("  [Perf] Mul:            %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 3.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 100.0);
}

TEST_F(ElementwisePerfTest, Scale_Bandwidth) {
    init_buffers();
    double ms = run_and_measure<kElementwiseScale>(
        d_out.data_, d_in0.data_, nullptr, PERF_N, 2.0f);
    // 读 1 + 写 1 = 2 * 128 MB = 256 MB
    double bw = (2.0 * BYTES) / (ms * 1e6);
    printf("  [Perf] Scale:          %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 2.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 100.0);
}

TEST_F(ElementwisePerfTest, Relu2_Bandwidth) {
    init_buffers();
    double ms = run_and_measure<kElementwiseRelu2>(
        d_out.data_, d_in0.data_, nullptr, PERF_N);
    double bw = (2.0 * BYTES) / (ms * 1e6);
    printf("  [Perf] Relu2:          %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 2.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 30.0);
}

TEST_F(ElementwisePerfTest, Silu_Bandwidth) {
    init_buffers();
    double ms = run_and_measure<kElementwiseSilu>(
        d_out.data_, d_in0.data_, nullptr, PERF_N);
    double bw = (2.0 * BYTES) / (ms * 1e6);
    printf("  [Perf] Silu:           %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 2.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 50.0);
}

TEST_F(ElementwisePerfTest, ClampSoftplus_Bandwidth) {
    init_buffers();
    auto d_bias = allocate_tensor<float>(TensorShape::make_1d(PERF_N));
    device_memset_zero(d_bias);
    double ms = run_and_measure<kElementwiseClampSoftplus>(
        d_out.data_, d_in0.data_, nullptr, PERF_N,
        1.f, d_bias.data_, 0.001f, 0.1f);
    // 读 in0 + bias + 写 out = 3 * 128 MB
    double bw = (3.0 * BYTES) / (ms * 1e6);
    printf("  [Perf] ClampSoftplus:  %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 3.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 30.0);
    free_tensor(d_bias);
}

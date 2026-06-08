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

/// BF16 版本的 run_and_measure
template<ElementwiseType Op>
double run_and_measure_bf16(bfloat16_t* d_out, bfloat16_t* d_in0, bfloat16_t* d_in1,
                             size_t size, float scale = 1.f,
                             bfloat16_t* d_bias = nullptr,
                             float clamp_min = 0.f, float clamp_max = 1.f,
                             int warmup_iters = 5, int bench_iters = 20) {
    cudaStream_t stream = 0;
    for (int i = 0; i < warmup_iters; ++i) {
        elementwise_ops_bf16<Op>(d_in0, d_in1, d_out, size,
                                  scale, d_bias, clamp_min, clamp_max, stream);
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        elementwise_ops_bf16<Op>(d_in0, d_in1, d_out, size,
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
// 3b. BF16 正确性测试
// ===========================================================================
// 注意：BF16 kernel 不做尾数处理，N 必须为 8 的倍数

class ElementwiseCorrectnessTestBF16 : public ::testing::Test {
protected:
    void SetUp() override { warmup_gpu(); }

    static constexpr size_t N = 1024;
};

static std::vector<bfloat16_t> float_to_bf16_vec(const std::vector<float>& src) {
    std::vector<bfloat16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = __float2bfloat16_rn(src[i]);
    return dst;
}

static std::vector<float> bf16_to_float_vec(const std::vector<bfloat16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = __bfloat162float(src[i]);
    return dst;
}

TEST_F(ElementwiseCorrectnessTestBF16, Add) {
    // 数据范围 [-4, 4] + [-2, 2] → [-6, 6]，BF16 步长 ≤ 0.125
    std::vector<float> in0(N), in1(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        float t = float(i) / (N - 1);
        in0[i] = t * 8.f - 4.f;
        in1[i] = t * 4.f - 2.f;
        expected[i] = in0[i] + in1[i];
    }
    auto in0_bf16 = float_to_bf16_vec(in0);
    auto in1_bf16 = float_to_bf16_vec(in1);
    auto d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_in1 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0_bf16.data());
    copy_host_to_device(d_in1, in1_bf16.data());

    elementwise_ops_bf16<kElementwiseAdd>(
        d_in0.data_, d_in1.data_, d_out.data_, N);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out_f, expected[i], 0.1f) << " at index " << i;
    }

    free_tensor(d_in0); free_tensor(d_in1); free_tensor(d_out);
}

TEST_F(ElementwiseCorrectnessTestBF16, Mul) {
    // 数据范围 [-2, 2] × [-1, 3.5] → [-7, 7]，乘法放大 BF16 量化误差
    std::vector<float> in0(N), in1(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        float t = float(i) / (N - 1);
        in0[i] = t * 4.f - 2.f;
        in1[i] = t * 3.f + 0.5f;
        expected[i] = in0[i] * in1[i];
    }
    auto in0_bf16 = float_to_bf16_vec(in0);
    auto in1_bf16 = float_to_bf16_vec(in1);
    auto d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_in1 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0_bf16.data());
    copy_host_to_device(d_in1, in1_bf16.data());

    elementwise_ops_bf16<kElementwiseMul>(
        d_in0.data_, d_in1.data_, d_out.data_, N);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out_f, expected[i], 0.5f) << " at index " << i;
    }

    free_tensor(d_in0); free_tensor(d_in1); free_tensor(d_out);
}

TEST_F(ElementwiseCorrectnessTestBF16, Scale) {
    const float alpha = 2.71828f;
    // 数据范围 [-3, 3] × 2.718 → [-8.15, 8.15]
    std::vector<float> in0(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        float t = float(i) / (N - 1);
        in0[i] = t * 6.f - 3.f;
        expected[i] = in0[i] * alpha;
    }
    auto in0_bf16 = float_to_bf16_vec(in0);
    auto d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0_bf16.data());

    elementwise_ops_bf16<kElementwiseScale>(
        d_in0.data_, nullptr, d_out.data_, N, alpha);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out_f, expected[i], 0.2f) << " at index " << i;
    }

    free_tensor(d_in0); free_tensor(d_out);
}

TEST_F(ElementwiseCorrectnessTestBF16, Relu2) {
    // 数据范围 [-4, 4] → relu² max = 16；BF16 步长在 16 附近 = 0.125
    std::vector<float> in0(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        float t = float(i) / (N - 1);
        in0[i] = t * 8.f - 4.f;
        expected[i] = ref::relu2(in0[i]);
    }
    auto in0_bf16 = float_to_bf16_vec(in0);
    auto d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0_bf16.data());

    elementwise_ops_bf16<kElementwiseRelu2>(
        d_in0.data_, nullptr, d_out.data_, N);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out_f, expected[i], 0.5f) << " at index " << i;
    }

    free_tensor(d_in0); free_tensor(d_out);
}

TEST_F(ElementwiseCorrectnessTestBF16, Silu) {
    // 数据范围 [-5, 5]；SiLU 在 ±5 处饱和，输出范围 ≈ [-0.27, 5]
    std::vector<float> in0(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        float t = float(i) / (N - 1);
        in0[i] = t * 10.f - 5.f;
        expected[i] = ref::silu(in0[i]);
    }
    auto in0_bf16 = float_to_bf16_vec(in0);
    auto d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0_bf16.data());

    elementwise_ops_bf16<kElementwiseSilu>(
        d_in0.data_, nullptr, d_out.data_, N);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        // SiLU 在 BF16 下包含 expf → float32 → BF16 两轮量化，0.2f 容忍
        EXPECT_NEAR(out_f, expected[i], 0.2f) << " at index " << i;
    }

    free_tensor(d_in0); free_tensor(d_out);
}

TEST_F(ElementwiseCorrectnessTestBF16, ClampSoftplus) {
    const float c_min = 0.001f, c_max = 0.1f;
    // 数据范围 [-3, 3]，偏置 [-0.48, 0.47]，输出被 clamp 到 [0.001, 0.1]
    std::vector<float> in0(N), bias(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        float t = float(i) / (N - 1);
        in0[i] = t * 6.f - 3.f;
        bias[i] = float(i % 96) * 0.01f - 0.48f;
        expected[i] = ref::clamp_softplus(in0[i], bias[i], c_min, c_max);
    }
    auto in0_bf16 = float_to_bf16_vec(in0);
    auto bias_bf16 = float_to_bf16_vec(bias);
    auto d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_bias = allocate_tensor<bfloat16_t>(TensorShape::make_1d(N));
    auto d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(N));
    copy_host_to_device(d_in0, in0_bf16.data());
    copy_host_to_device(d_bias, bias_bf16.data());

    elementwise_ops_bf16<kElementwiseClampSoftplus>(
        d_in0.data_, nullptr, d_out.data_, N,
        1.f, d_bias.data_, c_min, c_max);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(N);
    copy_device_to_host(out_bf16.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < N; ++i) {
        float out_f = __bfloat162float(out_bf16[i]);
        // 内核内部使用 FP32 计算，仅有输入/输出 BF16 量化误差（~0.002）
        EXPECT_NEAR(out_f, expected[i], 0.01f) << " at index " << i;
    }

    free_tensor(d_in0); free_tensor(d_bias); free_tensor(d_out);
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

// ===========================================================================
// 4b. BF16 性能基准测试
// ===========================================================================

class ElementwisePerfTestBF16 : public ::testing::Test {
protected:
    void SetUp() override { warmup_gpu(); }

    static constexpr size_t PERF_N = 32ULL * 1024 * 1024;
    static constexpr size_t BYTES  = PERF_N * sizeof(bfloat16_t);

    void init_buffers() {
        if (!d_in0.data_) {
            d_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(PERF_N));
        }
        if (!d_in1.data_) {
            d_in1 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(PERF_N));
        }
        if (!d_out.data_) {
            d_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(PERF_N));
        }
        h_buf.resize(PERF_N);
        for (size_t i = 0; i < PERF_N; ++i) h_buf[i] = float(i) * 0.001f - 5.f;
        std::vector<bfloat16_t> h_buf_bf16(PERF_N);
        for (size_t i = 0; i < PERF_N; ++i)
            h_buf_bf16[i] = __float2bfloat16_rn(h_buf[i]);
        copy_host_to_device(d_in0, h_buf_bf16.data());
        copy_host_to_device(d_in1, h_buf_bf16.data());
        cudaDeviceSynchronize();
    }

    Tensor<bfloat16_t> d_in0, d_in1, d_out;
    std::vector<float> h_buf;
};

TEST_F(ElementwisePerfTestBF16, Add_Bandwidth) {
    init_buffers();
    double ms = run_and_measure_bf16<kElementwiseAdd>(
        d_out.data_, d_in0.data_, d_in1.data_, PERF_N);
    double bw = (3.0 * BYTES) / (ms * 1e6);
    printf("  [PerfBF16] Add:            %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 3.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 100.0);
}

TEST_F(ElementwisePerfTestBF16, Mul_Bandwidth) {
    init_buffers();
    double ms = run_and_measure_bf16<kElementwiseMul>(
        d_out.data_, d_in0.data_, d_in1.data_, PERF_N);
    double bw = (3.0 * BYTES) / (ms * 1e6);
    printf("  [PerfBF16] Mul:            %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 3.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 100.0);
}

TEST_F(ElementwisePerfTestBF16, Scale_Bandwidth) {
    init_buffers();
    double ms = run_and_measure_bf16<kElementwiseScale>(
        d_out.data_, d_in0.data_, nullptr, PERF_N, 2.0f);
    double bw = (2.0 * BYTES) / (ms * 1e6);
    printf("  [PerfBF16] Scale:          %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 2.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 100.0);
}

TEST_F(ElementwisePerfTestBF16, Relu2_Bandwidth) {
    init_buffers();
    double ms = run_and_measure_bf16<kElementwiseRelu2>(
        d_out.data_, d_in0.data_, nullptr, PERF_N);
    double bw = (2.0 * BYTES) / (ms * 1e6);
    printf("  [PerfBF16] Relu2:          %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 2.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 30.0);
}

TEST_F(ElementwisePerfTestBF16, Silu_Bandwidth) {
    init_buffers();
    double ms = run_and_measure_bf16<kElementwiseSilu>(
        d_out.data_, d_in0.data_, nullptr, PERF_N);
    double bw = (2.0 * BYTES) / (ms * 1e6);
    printf("  [PerfBF16] Silu:           %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 2.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 50.0);
}

TEST_F(ElementwisePerfTestBF16, ClampSoftplus_Bandwidth) {
    init_buffers();
    auto d_bias = allocate_tensor<bfloat16_t>(TensorShape::make_1d(PERF_N));
    device_memset_zero(d_bias);
    double ms = run_and_measure_bf16<kElementwiseClampSoftplus>(
        d_out.data_, d_in0.data_, nullptr, PERF_N,
        1.f, d_bias.data_, 0.001f, 0.1f);
    double bw = (3.0 * BYTES) / (ms * 1e6);
    printf("  [PerfBF16] ClampSoftplus:  %6.3f ms | %7.2f GB/s | %uM elem (%.0f MB, read+write %.0f MB)\n",
           ms, bw, (unsigned)(PERF_N >> 20), BYTES / (1024.0 * 1024), 3.0 * BYTES / (1024.0 * 1024));
    EXPECT_GT(bw, 30.0);
    free_tensor(d_bias);
}

// ===========================================================================
// 5. FP32 vs BF16 速度对比
// ===========================================================================

class ElementwiseComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        warmup_gpu();
        // 分配 FP32 buffer
        if (!fp32_in0.data_) {
            fp32_in0 = allocate_tensor<float>(TensorShape::make_1d(PERF_N));
            fp32_in1 = allocate_tensor<float>(TensorShape::make_1d(PERF_N));
            fp32_out = allocate_tensor_zeros<float>(TensorShape::make_1d(PERF_N));
        }
        // 分配 BF16 buffer
        if (!bf16_in0.data_) {
            bf16_in0 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(PERF_N));
            bf16_in1 = allocate_tensor<bfloat16_t>(TensorShape::make_1d(PERF_N));
            bf16_out = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(PERF_N));
        }
        // 填充数据
        h_buf.resize(PERF_N);
        for (size_t i = 0; i < PERF_N; ++i) h_buf[i] = float(i) * 0.001f - 5.f;
        std::vector<bfloat16_t> h_buf_bf16(PERF_N);
        for (size_t i = 0; i < PERF_N; ++i)
            h_buf_bf16[i] = __float2bfloat16_rn(h_buf[i]);
        copy_host_to_device(fp32_in0, h_buf.data());
        copy_host_to_device(fp32_in1, h_buf.data());
        copy_host_to_device(bf16_in0, h_buf_bf16.data());
        copy_host_to_device(bf16_in1, h_buf_bf16.data());
        cudaDeviceSynchronize();
    }

    template<ElementwiseType Op>
    void run_comparison(const char* op_name,
                        float scale = 1.f,
                        bool use_second_input = false,
                        float* fp32_bias = nullptr,
                        bfloat16_t* bf16_bias = nullptr,
                        float clamp_min = 0.f,
                        float clamp_max = 1.f) {
        // FP32
        double fp32_ms = run_and_measure<Op>(
            fp32_out.data_, fp32_in0.data_,
            use_second_input ? fp32_in1.data_ : nullptr,
            PERF_N, scale, fp32_bias, clamp_min, clamp_max);
        // BF16
        double bf16_ms = run_and_measure_bf16<Op>(
            bf16_out.data_, bf16_in0.data_,
            use_second_input ? bf16_in1.data_ : nullptr,
            PERF_N, scale, bf16_bias, clamp_min, clamp_max);

        double speedup = fp32_ms / bf16_ms;
        double fp32_bytes = ((use_second_input ? 2.0 : 1.0) + (fp32_bias ? 1.0 : 0.0) + 1.0)
                            * PERF_N * sizeof(float);
        double bf16_bytes = ((use_second_input ? 2.0 : 1.0) + (bf16_bias ? 1.0 : 0.0) + 1.0)
                            * PERF_N * sizeof(bfloat16_t);
        double fp32_bw = fp32_bytes / (fp32_ms * 1e6);
        double bf16_bw = bf16_bytes / (bf16_ms * 1e6);

        printf("  [Compare] %-16s FP32: %6.3f ms (%5.1f GB/s) | BF16: %6.3f ms (%5.1f GB/s) | Speedup: %.2f x\n",
               op_name, fp32_ms, fp32_bw, bf16_ms, bf16_bw, speedup);

        // BF16 带宽不应低于 FP32 的一半（退化检查）
        EXPECT_GT(bf16_bw, fp32_bw * 0.4);
    }

    static constexpr size_t PERF_N = 32ULL * 1024 * 1024;

    Tensor<float> fp32_in0, fp32_in1, fp32_out;
    Tensor<bfloat16_t> bf16_in0, bf16_in1, bf16_out;
    std::vector<float> h_buf;
};

TEST_F(ElementwiseComparisonTest, Add_Speedup) {
    run_comparison<kElementwiseAdd>("Add", 1.f, true);
}

TEST_F(ElementwiseComparisonTest, Mul_Speedup) {
    run_comparison<kElementwiseMul>("Mul", 1.f, true);
}

TEST_F(ElementwiseComparisonTest, Scale_Speedup) {
    run_comparison<kElementwiseScale>("Scale", 2.0f, false);
}

TEST_F(ElementwiseComparisonTest, Relu2_Speedup) {
    run_comparison<kElementwiseRelu2>("Relu2", 1.f, false);
}

TEST_F(ElementwiseComparisonTest, Silu_Speedup) {
    run_comparison<kElementwiseSilu>("Silu", 1.f, false);
}

TEST_F(ElementwiseComparisonTest, ClampSoftplus_Speedup) {
    auto fp32_bias = allocate_tensor<float>(TensorShape::make_1d(PERF_N));
    auto bf16_bias = allocate_tensor<bfloat16_t>(TensorShape::make_1d(PERF_N));
    device_memset_zero(fp32_bias);
    device_memset_zero(bf16_bias);
    run_comparison<kElementwiseClampSoftplus>(
        "ClampSoftplus", 1.f, false,
        fp32_bias.data_, bf16_bias.data_, 0.001f, 0.1f);
    free_tensor(fp32_bias);
    free_tensor(bf16_bias);
}

// ===========================================================================
// test_reduce.cu — 规约算子测试（正确性 + 性能）
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/reduce.cuh"

using namespace nemotron;
using namespace nemotron::ops;

// ===========================================================================
// 1. CPU 参考实现
// ===========================================================================
namespace ref {
    void rmsnorm_fp32(const float *x, float *y, const float *w, size_t n, float eps) {
        float sum_sq = 0.f;
        for (size_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
        float scale = 1.f / std::sqrt(sum_sq / n + eps);
        for (size_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
    }

    void rmsnorm_bf16(const float *x, float *y, const float *w, size_t n, float eps) {
        // 模拟 bf16 量化: float → bf16 → float
        std::vector<float> x_bf16(n);
        for (size_t i = 0; i < n; ++i) {
            x_bf16[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
        }
        float sum_sq = 0.f;
        for (size_t i = 0; i < n; ++i) sum_sq += x_bf16[i] * x_bf16[i];
        float scale = 1.f / std::sqrt(sum_sq / n + eps);
        for (size_t i = 0; i < n; ++i) {
            float val = x_bf16[i] * scale * __bfloat162float(__float2bfloat16_rn(w[i]));
            y[i] = __bfloat162float(__float2bfloat16_rn(val));
        }
    }

    inline float silu_ref(float z) { return z / (1.f + std::exp(-z)); }

    // 对齐 HF MambaRMSNormGated(norm_before_gate=False)：
    //   gated = x*silu(gate); rstd = rsqrt(mean_group(gated^2)+eps); y = gated*rstd*w
    void rmsnorm_gated_fp32(
        const float *x, float *y, const float *w, const float *gate,
        size_t cols, size_t group_size, float eps
    ) {
        size_t groups = cols / group_size;
        for (size_t g = 0; g < groups; ++g) {
            float sum_sq = 0.f;
            for (size_t i = g * group_size; i < (g + 1) * group_size; ++i) {
                float gated = x[i] * silu_ref(gate[i]);
                sum_sq += gated * gated;
            }
            float scale = 1.f / std::sqrt(sum_sq / group_size + eps);
            for (size_t i = g * group_size; i < (g + 1) * group_size; ++i)
                y[i] = x[i] * silu_ref(gate[i]) * scale * w[i];
        }
    }

    void rmsnorm_gated_bf16(
        const float *x, float *y, const float *w, const float *gate,
        size_t cols, size_t group_size, float eps
    ) {
        std::vector<float> x_bf16(cols), gate_bf16(cols);
        for (size_t i = 0; i < cols; ++i) {
            x_bf16[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
            gate_bf16[i] = __bfloat162float(__float2bfloat16_rn(gate[i]));
        }
        size_t groups = cols / group_size;
        for (size_t g = 0; g < groups; ++g) {
            float sum_sq = 0.f;
            for (size_t i = g * group_size; i < (g + 1) * group_size; ++i) {
                float gated = x_bf16[i] * silu_ref(gate_bf16[i]);
                sum_sq += gated * gated;
            }
            float scale = 1.f / std::sqrt(sum_sq / group_size + eps);
            for (size_t i = g * group_size; i < (g + 1) * group_size; ++i) {
                float gated = x_bf16[i] * silu_ref(gate_bf16[i]);
                float v = gated * scale * __bfloat162float(__float2bfloat16_rn(w[i]));
                y[i] = __bfloat162float(__float2bfloat16_rn(v));
            }
        }
    }
} // namespace ref

// ===========================================================================
// 2. 辅助函数
// ===========================================================================

static void warmup_gpu() {
    float *buf = nullptr;
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
// 3. FP32 RMSNorm 正确性测试
// ===========================================================================

class RMSNormCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset(); // 清理前面测试套件累积的显存
        warmup_gpu();
    }
};

TEST_F(RMSNormCorrectnessTest, FP32_Small) {
    const size_t rows = 4, cols = 256;

    auto x = rand_vec(rows * cols, 5.f);
    auto w = rand_vec(cols, 2.f);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_fp32(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), cols, 1e-5f
        );

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_w, w.data());
    cudaDeviceSynchronize();

    rmsnorm_fp32(d_x.data_, d_y.data_, d_w.data_, rows, cols, 1e-5f);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTest, FP32_FullHidden) {
    // 模拟真实模型维度: [B=2, S=128, hidden=3136]
    const size_t rows = 256, cols = 3136;

    auto x = rand_vec(rows * cols);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_fp32(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), cols, 1e-5f
        );

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_w, w.data());
    cudaDeviceSynchronize();

    rmsnorm_fp32(d_x.data_, d_y.data_, d_w.data_, rows, cols, 1e-5f);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTest, FP32_GridStride) {
    // grid-stride loop: rows > gridDim.x 时验证正确性
    const size_t rows = 128, cols = 1024;
    const size_t grid = 4;

    auto x = rand_vec(rows * cols);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_fp32(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), cols, 1e-5f
        );

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_w, w.data());
    cudaDeviceSynchronize();

    constexpr int block_dimx = 256;
    rmsnorm_launch_fp32<<<grid, block_dimx>>>(d_x.data_, d_y.data_, d_w.data_, rows, cols, 1e-5f);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTest, FP32_SingleRow) {
    const size_t rows = 1, cols = 3136;

    auto x = rand_vec(rows * cols);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    ref::rmsnorm_fp32(x.data(), expected.data(), w.data(), cols, 1e-5f);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_w, w.data());
    cudaDeviceSynchronize();

    rmsnorm_fp32(d_x.data_, d_y.data_, d_w.data_, rows, cols, 1e-5f);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
}

// ===========================================================================
// 4. BF16 RMSNorm 正确性测试
// ===========================================================================

class RMSNormCorrectnessTestBF16 : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(RMSNormCorrectnessTestBF16, Small) {
    const size_t rows = 4, cols = 256;

    auto x = rand_vec(rows * cols, 3.f);
    auto w = rand_vec(cols, 1.5f);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_bf16(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), cols, 1e-5f
        );

    std::vector<bfloat16_t> x_bf16(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
        x_bf16[i] = __float2bfloat16_rn(x[i]);

    auto d_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    copy_host_to_device(d_x, x_bf16.data());
    copy_host_to_device(d_w, w.data());
    cudaDeviceSynchronize();

    rmsnorm_bf16(d_x.data_, d_y.data_, d_w.data_, rows, cols, 1e-5f);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(rows * cols);
    copy_device_to_host(out_bf16.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i) {
        out[i] = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out[i], expected[i], 0.5f) << " at index " << i;
    }

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTestBF16, FullHidden) {
    const size_t rows = 32, cols = 3136;

    auto x = rand_vec(rows * cols, 2.f);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_bf16(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), cols, 1e-5f
        );

    std::vector<bfloat16_t> x_bf16(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
        x_bf16[i] = __float2bfloat16_rn(x[i]);

    auto d_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    copy_host_to_device(d_x, x_bf16.data());
    copy_host_to_device(d_w, w.data());
    cudaDeviceSynchronize();

    rmsnorm_bf16(d_x.data_, d_y.data_, d_w.data_, rows, cols, 1e-5f);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(rows * cols);
    copy_device_to_host(out_bf16.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i) {
        out[i] = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out[i], expected[i], 0.5f) << " at index " << i;
    }

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
}

// ===========================================================================
// 5. FP32 性能基准测试
// ===========================================================================

class RMSNormPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }

    static constexpr size_t ROWS = 8192;
    static constexpr size_t COLS = 3136;

    void init_data() {
        if (!d_x.data_) d_x = allocate_tensor<float>(TensorShape::make_1d(ROWS * COLS));
        if (!d_y.data_) d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(ROWS * COLS));
        if (!d_w.data_) d_w = allocate_tensor<float>(TensorShape::make_1d(COLS));
        h_buf.resize(ROWS * COLS);
        for (size_t i = 0; i < ROWS * COLS; ++i) h_buf[i] = float(i) * 0.001f - 5.f;
        copy_host_to_device(d_x, h_buf.data());
        copy_host_to_device(d_w, h_buf.data());
        cudaDeviceSynchronize();
    }

    Tensor<float> d_x, d_y, d_w;
    std::vector<float> h_buf;
};

TEST_F(RMSNormPerfTest, FP32_Bandwidth) {
    init_data();

    int warmup_iters = 5, bench_iters = 20;
    for (int i = 0; i < warmup_iters; ++i) {
        rmsnorm_fp32(d_x.data_, d_y.data_, d_w.data_, ROWS, COLS, 1e-5f);
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        rmsnorm_fp32(d_x.data_, d_y.data_, d_w.data_, ROWS, COLS, 1e-5f);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    size_t bytes = ROWS * COLS * sizeof(float);
    double bw = (bytes * 2.0) / (ms * 1e6);
    printf(
        "  [Perf] RMSNorm FP32:   %6.3f ms | %7.2f GB/s | %zu rows x %zu cols\n",
        ms, bw, (size_t) ROWS, (size_t) COLS
    );
    EXPECT_GT(bw, 50.0);
}

// ===========================================================================
// 6. BF16 性能基准测试
// ===========================================================================

class RMSNormPerfTestBF16 : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }

    static constexpr size_t ROWS = 8192;
    static constexpr size_t COLS = 3136;

    void init_data() {
        if (!d_x.data_) d_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(ROWS * COLS));
        if (!d_y.data_) d_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(ROWS * COLS));
        if (!d_w.data_) d_w = allocate_tensor<float>(TensorShape::make_1d(COLS));

        std::vector<float> h_float(ROWS * COLS);
        for (size_t i = 0; i < ROWS * COLS; ++i) h_float[i] = float(i) * 0.001f - 5.f;
        std::vector<bfloat16_t> h_bf16(ROWS * COLS);
        for (size_t i = 0; i < ROWS * COLS; ++i)
            h_bf16[i] = __float2bfloat16_rn(h_float[i]);

        copy_host_to_device(d_x, h_bf16.data());
        copy_host_to_device(d_w, h_float.data());
        cudaDeviceSynchronize();
    }

    Tensor<bfloat16_t> d_x, d_y;
    Tensor<float> d_w;
};

TEST_F(RMSNormPerfTestBF16, Bandwidth) {
    init_data();

    int warmup_iters = 5, bench_iters = 20;
    for (int i = 0; i < warmup_iters; ++i) {
        rmsnorm_bf16(d_x.data_, d_y.data_, d_w.data_, ROWS, COLS, 1e-5f);
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        rmsnorm_bf16(d_x.data_, d_y.data_, d_w.data_, ROWS, COLS, 1e-5f);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    size_t bytes = ROWS * COLS * sizeof(bfloat16_t);
    double bw = (bytes * 2.0) / (ms * 1e6);
    printf(
        "  [Perf] RMSNorm BF16:   %6.3f ms | %7.2f GB/s | %zu rows x %zu cols\n",
        ms, bw, (size_t) ROWS, (size_t) COLS
    );
    EXPECT_GT(bw, 50.0);
}

// ===========================================================================
// 7. RMSNorm Gated 正确性测试
// ===========================================================================

class RMSNormGatedCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(RMSNormGatedCorrectnessTest, FP32_Small) {
    // Mamba 实际配置: 7680 分 8 组，每组 960，这里用小尺寸验证
    const size_t rows = 4, cols = 192, group_size = 64, groups = cols / group_size;

    auto x = rand_vec(rows * cols, 3.f);
    auto gate = rand_vec(rows * cols, 2.f);
    auto w = rand_vec(cols, 1.5f);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_gated_fp32(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), gate.data() + r * cols, cols, group_size, 1e-5f
        );

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    auto d_gate = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_w, w.data());
    copy_host_to_device(d_gate, gate.data());
    cudaDeviceSynchronize();

    rmsnorm_gated_fp32<4>(
        d_x.data_, d_y.data_, d_w.data_, d_gate.data_,
        rows, cols, group_size, 1e-5f
    );
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
    free_tensor(d_gate);
}

TEST_F(RMSNormGatedCorrectnessTest, FP32_FullMamba) {
    // Mamba 真实维度: 7680, group=960, groups=8
    const size_t rows = 4096, cols = 7680, group_size = 960;

    auto x = rand_vec(rows * cols);
    auto gate = rand_vec(rows * cols);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_gated_fp32(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), gate.data() + r * cols, cols, group_size, 1e-5f
        );

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    auto d_gate = allocate_tensor<float>(TensorShape::make_1d(rows * cols));
    copy_host_to_device(d_x, x.data());
    copy_host_to_device(d_w, w.data());
    copy_host_to_device(d_gate, gate.data());
    cudaDeviceSynchronize();

    rmsnorm_gated_fp32<8>(
        d_x.data_, d_y.data_, d_w.data_, d_gate.data_,
        rows, cols, group_size, 1e-5f
    );
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-5f) << " at index " << i;

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
    free_tensor(d_gate);
}

TEST_F(RMSNormGatedCorrectnessTest, BF16_FullMamba) {
    const size_t rows = 4096, cols = 7680, group_size = 960;

    auto x = rand_vec(rows * cols, 2.f);
    auto gate = rand_vec(rows * cols, 1.5f);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_gated_bf16(
            x.data() + r * cols, expected.data() + r * cols,
            w.data(), gate.data() + r * cols, cols, group_size, 1e-5f
        );

    std::vector<bfloat16_t> x_bf16(rows * cols), gate_bf16(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        x_bf16[i] = __float2bfloat16_rn(x[i]);
        gate_bf16[i] = __float2bfloat16_rn(gate[i]);
    }

    auto d_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(rows * cols));
    auto d_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(rows * cols));
    auto d_w = allocate_tensor<float>(TensorShape::make_1d(cols));
    auto d_gate = allocate_tensor<bfloat16_t>(TensorShape::make_1d(rows * cols));
    copy_host_to_device(d_x, x_bf16.data());
    copy_host_to_device(d_w, w.data());
    copy_host_to_device(d_gate, gate_bf16.data());
    cudaDeviceSynchronize();

    rmsnorm_gated_bf16<8>(
        d_x.data_, d_y.data_, d_w.data_, d_gate.data_,
        rows, cols, group_size, 1e-5f
    );
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(rows * cols);
    copy_device_to_host(out_bf16.data(), d_y);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < rows * cols; ++i) {
        out[i] = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out[i], expected[i], 0.5f) << " at index " << i;
    }

    free_tensor(d_x);
    free_tensor(d_y);
    free_tensor(d_w);
    free_tensor(d_gate);
}

// ===========================================================================
// 8. FP32 vs BF16 门控 RMSNorm 速度对比
// ===========================================================================

class RMSNormGatedComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!fp32_x.data_) {
            fp32_x = allocate_tensor<float>(TensorShape::make_1d(ROWS * COLS));
            fp32_y = allocate_tensor_zeros<float>(TensorShape::make_1d(ROWS * COLS));
            fp32_gate = allocate_tensor<float>(TensorShape::make_1d(ROWS * COLS));
            fp32_w = allocate_tensor<float>(TensorShape::make_1d(COLS));
        }
        if (!bf16_x.data_) {
            bf16_x = allocate_tensor<bfloat16_t>(TensorShape::make_1d(ROWS * COLS));
            bf16_y = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(ROWS * COLS));
            bf16_gate = allocate_tensor<bfloat16_t>(TensorShape::make_1d(ROWS * COLS));
            bf16_w = allocate_tensor<float>(TensorShape::make_1d(COLS));
        }

        h_buf.resize(ROWS * COLS);
        for (size_t i = 0; i < ROWS * COLS; ++i) h_buf[i] = float(i) * 0.001f - 5.f;
        std::vector<bfloat16_t> h_buf_bf16(ROWS * COLS);
        for (size_t i = 0; i < ROWS * COLS; ++i)
            h_buf_bf16[i] = __float2bfloat16_rn(h_buf[i]);

        copy_host_to_device(fp32_x, h_buf.data());
        copy_host_to_device(fp32_gate, h_buf.data());
        copy_host_to_device(fp32_w, h_buf.data());
        copy_host_to_device(bf16_x, h_buf_bf16.data());
        copy_host_to_device(bf16_gate, h_buf_bf16.data());
        copy_host_to_device(bf16_w, h_buf.data());
        cudaDeviceSynchronize();
    }

    static constexpr size_t ROWS = 4096;
    static constexpr size_t COLS = 7680;
    static constexpr size_t GROUP_SIZE = 960;

    Tensor<float> fp32_x, fp32_y, fp32_w, fp32_gate;
    Tensor<bfloat16_t> bf16_x, bf16_y, bf16_gate;
    Tensor<float> bf16_w;
    std::vector<float> h_buf;
};

TEST_F(RMSNormGatedComparisonTest, Speedup) {
    int warmup_iters = 5, bench_iters = 20;

    // FP32
    for (int i = 0; i < warmup_iters; ++i)
        rmsnorm_gated_fp32<8>(
            fp32_x.data_, fp32_y.data_, fp32_w.data_, fp32_gate.data_,
            ROWS, COLS, GROUP_SIZE, 1e-5f
        );
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        rmsnorm_gated_fp32<8>(
            fp32_x.data_, fp32_y.data_, fp32_w.data_, fp32_gate.data_,
            ROWS, COLS, GROUP_SIZE, 1e-5f
        );
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    // BF16
    for (int i = 0; i < warmup_iters; ++i)
        rmsnorm_gated_bf16<8>(
            bf16_x.data_, bf16_y.data_, bf16_w.data_, bf16_gate.data_,
            ROWS, COLS, GROUP_SIZE, 1e-5f
        );
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        rmsnorm_gated_bf16<8>(
            bf16_x.data_, bf16_y.data_, bf16_w.data_, bf16_gate.data_,
            ROWS, COLS, GROUP_SIZE, 1e-5f
        );
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    double speedup = fp32_ms / bf16_ms;
    size_t bytes_per_elem = 4; // read x + gate + write y = 3 × sizeof = 12 for FP32, 6 for BF16
    size_t fp32_bytes = ROWS * COLS * bytes_per_elem * 3;
    size_t bf16_bytes = ROWS * COLS * (bytes_per_elem / 2) * 3;
    double fp32_bw = fp32_bytes / (fp32_ms * 1e6);
    double bf16_bw = bf16_bytes / (bf16_ms * 1e6);

    printf(
        "  [Compare] RMSNorm Gated       FP32: %6.3f ms (%5.1f GB/s) | BF16: %6.3f ms (%5.1f GB/s) | Speedup: %.2f x\n",
        fp32_ms, fp32_bw, bf16_ms, bf16_bw, speedup
    );
    EXPECT_GT(speedup, 0.5);
}

// ===========================================================================
// 9. Segmented Cumsum 正确性 + 性能测试
// ===========================================================================

namespace ref {

void segment_sum(const float* A_cum, float* L, size_t chunk_size, size_t num_triplets) {
    for (size_t n = 0; n < num_triplets; ++n) {
        const float* A = A_cum + n * chunk_size;
        float* L_n = L + n * chunk_size * chunk_size;
        for (size_t t = 0; t < chunk_size; ++t) {
            float A_t = A[t];
            for (size_t s = 0; s < chunk_size; ++s)
                L_n[t * chunk_size + s] = (s <= t) ? std::exp(A_t - A[s]) : 0.f;
        }
    }
}

}  // namespace ref

class SegmentedCumsumTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(SegmentedCumsumTest, FP32_Small) {
    constexpr size_t L = 8, M = 2;  // 2 组 triplets, 每组 chunk_size=8
    const size_t NN = M * L;

    // 生成 log-space 衰减因子 [-3, 0]
    std::vector<float> A_cum(NN);
    for (size_t i = 0; i < NN; ++i)
        A_cum[i] = -float(i % L) * 0.2f;  // 递减的 cumsum

    std::vector<float> expected(M * L * L), out(M * L * L);
    ref::segment_sum(A_cum.data(), expected.data(), L, M);

    auto d_A = allocate_tensor<float>(TensorShape::make_1d(NN));
    auto d_L = allocate_tensor_zeros<float>(TensorShape::make_1d(M * L * L));
    copy_host_to_device(d_A, A_cum.data());
    cudaDeviceSynchronize();

    segment_sum_fp32<L>(d_A.data_, d_L.data_, M);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_L);
    cudaDeviceSynchronize();

    for (size_t n = 0; n < M; ++n) {
        // 验证下三角结构
        for (size_t t = 0; t < L; ++t) {
            for (size_t s = 0; s < L; ++s) {
                size_t idx = n * L * L + t * L + s;
                if (s > t) {
                    EXPECT_FLOAT_EQ(out[idx], 0.f) << " upper-tri should be 0 at t=" << t << " s=" << s;
                } else {
                    EXPECT_NEAR(out[idx], expected[idx], 1e-5f) << " at triplet=" << n << " t=" << t << " s=" << s;
                }
            }
        }
    }

    free_tensor(d_A); free_tensor(d_L);
}

TEST_F(SegmentedCumsumTest, FP32_FullChunk) {
    // Mamba 真实 chunk_size=256, 4 组 triplets
    constexpr size_t L = 256, M = 4;
    const size_t NN = M * L;

    std::vector<float> A_cum(NN);
    for (size_t i = 0; i < NN; ++i)
        A_cum[i] = float(i % L) * -0.05f;  // 递减, 到结尾 exp(-12.75) ≈ 3e-6

    std::vector<float> expected(M * L * L), out(M * L * L);
    ref::segment_sum(A_cum.data(), expected.data(), L, M);

    auto d_A = allocate_tensor<float>(TensorShape::make_1d(NN));
    auto d_L = allocate_tensor_zeros<float>(TensorShape::make_1d(M * L * L));
    copy_host_to_device(d_A, A_cum.data());
    cudaDeviceSynchronize();

    segment_sum_fp32<L>(d_A.data_, d_L.data_, M);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_L);
    cudaDeviceSynchronize();

    // 抽样验证最上面几行和尾行，不全量 65536 元素
    size_t n = 0;
    for (size_t t : {0, 1, 2, 127, 128, 254, 255}) {
        // 验证对角线 exp(0) ≈ 1
        size_t diag = n * L * L + t * L + t;
        EXPECT_NEAR(out[diag], 1.0f, 1e-5f) << " diag t=" << t;
        // 验证上三角全是 0
        if (t < L - 1) {
            size_t upper = n * L * L + t * L + t + 1;
            EXPECT_FLOAT_EQ(out[upper], 0.f) << " upper at t=" << t;
        }
    }

    free_tensor(d_A); free_tensor(d_L);
}

// ===========================================================================
// 10. Segmented Cumsum 性能基准
// ===========================================================================

class SegmentedCumsumPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!d_A.data_) {
            d_A = allocate_tensor<float>(TensorShape::make_1d(M * L));
            d_L = allocate_tensor_zeros<float>(TensorShape::make_1d(M * L * L));
        }

        h_buf.resize(M * L);
        for (size_t i = 0; i < M * L; ++i)
            h_buf[i] = -float(i % L) * 0.05f;
        copy_host_to_device(d_A, h_buf.data());
        cudaDeviceSynchronize();
    }

    static constexpr size_t L = 256;
    static constexpr size_t M = 128;  // 128 个独立 cumsum, 模拟 B=1,H=8,C=16
    Tensor<float> d_A, d_L;
    std::vector<float> h_buf;
};

TEST_F(SegmentedCumsumPerfTest, FP32_Bandwidth) {
    int warmup_iters = 3, bench_iters = 10;
    for (int i = 0; i < warmup_iters; ++i)
        segment_sum_fp32<L>(d_A.data_, d_L.data_, M);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        segment_sum_fp32<L>(d_A.data_, d_L.data_, M);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    size_t write_bytes = M * L * L * sizeof(float);  // 写整个 L 矩阵
    size_t read_bytes = M * L * sizeof(float);        // 读 A_cum
    double bw = (write_bytes + read_bytes) / (ms * 1e6);
    printf("  [Perf] SegmentSum FP32:  %6.3f ms | %7.2f GB/s | %zu triplets x %zu x %zu (%zu MB out)\n",
           ms, bw, (size_t)M, (size_t)L, (size_t)L, write_bytes / (1024 * 1024));

    // 只在 FP32 测试里验证计算带宽合理（expf 密集）
    EXPECT_GT(bw, 10.0);
}

// ===========================================================================
// 11. BF16 Segmented Cumsum 正确性
// ===========================================================================

namespace ref {

void segment_sum_bf16(const float* A_cum_fp32, float* L, size_t chunk_size, size_t num_triplets) {
    // 模拟 BF16 IO: cumsum 读入时量化一次，exp 在 FP32 算，输出再量化一次
    for (size_t n = 0; n < num_triplets; ++n) {
        const float* A = A_cum_fp32 + n * chunk_size;
        float* L_n = L + n * chunk_size * chunk_size;
        for (size_t t = 0; t < chunk_size; ++t) {
            float A_t = __bfloat162float(__float2bfloat16_rn(A[t]));
            for (size_t s = 0; s < chunk_size; ++s) {
                if (s <= t) {
                    float diff = A_t - __bfloat162float(__float2bfloat16_rn(A[s]));
                    L_n[t * chunk_size + s] = __bfloat162float(__float2bfloat16_rn(expf(diff)));
                } else {
                    L_n[t * chunk_size + s] = 0.f;
                }
            }
        }
    }
}

}  // namespace ref

class SegmentedCumsumBF16Test : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(SegmentedCumsumBF16Test, Small) {
    constexpr size_t L = 8, M = 2;

    std::vector<float> A_cum(M * L);
    for (size_t i = 0; i < M * L; ++i)
        A_cum[i] = -float(i % L) * 0.2f;

    std::vector<float> expected(M * L * L), out_fp32(M * L * L);
    ref::segment_sum_bf16(A_cum.data(), expected.data(), L, M);

    std::vector<bfloat16_t> A_bf16(M * L);
    for (size_t i = 0; i < M * L; ++i)
        A_bf16[i] = __float2bfloat16_rn(A_cum[i]);

    auto d_A = allocate_tensor<bfloat16_t>(TensorShape::make_1d(M * L));
    auto d_L = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(M * L * L));
    copy_host_to_device(d_A, A_bf16.data());
    cudaDeviceSynchronize();

    segment_sum_bf16<L>(d_A.data_, d_L.data_, M);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(M * L * L);
    copy_device_to_host(out_bf16.data(), d_L);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < M * L * L; ++i) {
        out_fp32[i] = __bfloat162float(out_bf16[i]);
        EXPECT_NEAR(out_fp32[i], expected[i], 0.01f) << " at index " << i;
    }

    free_tensor(d_A); free_tensor(d_L);
}

TEST_F(SegmentedCumsumBF16Test, FullChunk) {
    constexpr size_t L = 256, M = 4;

    std::vector<float> A_cum(M * L);
    for (size_t i = 0; i < M * L; ++i)
        A_cum[i] = float(i % L) * -0.05f;

    std::vector<float> expected(M * L * L);
    ref::segment_sum_bf16(A_cum.data(), expected.data(), L, M);

    std::vector<bfloat16_t> A_bf16(M * L);
    for (size_t i = 0; i < M * L; ++i)
        A_bf16[i] = __float2bfloat16_rn(A_cum[i]);

    auto d_A = allocate_tensor<bfloat16_t>(TensorShape::make_1d(M * L));
    auto d_L = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(M * L * L));
    copy_host_to_device(d_A, A_bf16.data());
    cudaDeviceSynchronize();

    segment_sum_bf16<L>(d_A.data_, d_L.data_, M);
    cudaDeviceSynchronize();

    std::vector<bfloat16_t> out_bf16(M * L * L);
    copy_device_to_host(out_bf16.data(), d_L);
    cudaDeviceSynchronize();

    // 验证下三角结构
    size_t n = 0;
    for (size_t t : {0, 1, 127, 255}) {
        size_t diag = n * L * L + t * L + t;
        float diag_val = __bfloat162float(out_bf16[diag]);
        EXPECT_NEAR(diag_val, 1.0f, 0.05f) << " diag t=" << t;
        if (t < L - 1) {
            size_t upper = n * L * L + t * L + t + 1;
            EXPECT_FLOAT_EQ(__bfloat162float(out_bf16[upper]), 0.f) << " upper at t=" << t;
        }
    }

    free_tensor(d_A); free_tensor(d_L);
}

// ===========================================================================
// 12. BF16 Segmented Cumsum 性能 + FP32 vs BF16 对比
// ===========================================================================

class SegmentedCumsumComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!fp32_A.data_) {
            fp32_A = allocate_tensor<float>(TensorShape::make_1d(M * L));
            fp32_L = allocate_tensor_zeros<float>(TensorShape::make_1d(M * L * L));
        }
        if (!bf16_A.data_) {
            bf16_A = allocate_tensor<bfloat16_t>(TensorShape::make_1d(M * L));
            bf16_L = allocate_tensor_zeros<bfloat16_t>(TensorShape::make_1d(M * L * L));
        }

        h_fp32.resize(M * L);
        for (size_t i = 0; i < M * L; ++i)
            h_fp32[i] = -float(i % L) * 0.05f;
        std::vector<bfloat16_t> h_bf16(M * L);
        for (size_t i = 0; i < M * L; ++i)
            h_bf16[i] = __float2bfloat16_rn(h_fp32[i]);

        copy_host_to_device(fp32_A, h_fp32.data());
        copy_host_to_device(bf16_A, h_bf16.data());
        cudaDeviceSynchronize();
    }

    static constexpr size_t L = 256;
    static constexpr size_t M = 1024;
    Tensor<float> fp32_A, fp32_L;
    Tensor<bfloat16_t> bf16_A, bf16_L;
    std::vector<float> h_fp32;
};

TEST_F(SegmentedCumsumComparisonTest, Speedup) {
    int warmup_iters = 3, bench_iters = 10;

    // FP32
    for (int i = 0; i < warmup_iters; ++i)
        segment_sum_fp32<L>(fp32_A.data_, fp32_L.data_, M);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        segment_sum_fp32<L>(fp32_A.data_, fp32_L.data_, M);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    // BF16
    for (int i = 0; i < warmup_iters; ++i)
        segment_sum_bf16<L>(bf16_A.data_, bf16_L.data_, M);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        segment_sum_bf16<L>(bf16_A.data_, bf16_L.data_, M);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    double speedup = fp32_ms / bf16_ms;
    size_t fp32_traffic = M * L * L * sizeof(float) + M * L * sizeof(float);
    size_t bf16_traffic = M * L * L * sizeof(bfloat16_t) + M * L * sizeof(bfloat16_t);
    double fp32_bw = fp32_traffic / (fp32_ms * 1e6);
    double bf16_bw = bf16_traffic / (bf16_ms * 1e6);

    printf("  [Compare] SegmentSum       FP32: %6.3f ms (%5.1f GB/s) | BF16: %6.3f ms (%5.1f GB/s) | Speedup: %.2f x\n",
           fp32_ms, fp32_bw, bf16_ms, bf16_bw, speedup);
    EXPECT_GT(speedup, 0.5);
}

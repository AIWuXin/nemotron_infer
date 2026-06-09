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

void rmsnorm_fp32(const float* x, float* y, const float* w, size_t n, float eps) {
    float sum_sq = 0.f;
    for (size_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float scale = 1.f / std::sqrt(sum_sq / n + eps);
    for (size_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

void rmsnorm_bf16(const float* x, float* y, const float* w, size_t n, float eps) {
    // 模拟 bf16 量化: float → bf16 → float
    std::vector<float> x_bf16(n), w_bf16(n);
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

}  // namespace ref

// ===========================================================================
// 2. 辅助函数
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
// 3. FP32 RMSNorm 正确性测试
// ===========================================================================

class RMSNormCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();  // 清理前面测试套件累积的显存
        warmup_gpu();
    }
};

TEST_F(RMSNormCorrectnessTest, FP32_Small) {
    const size_t rows = 4, cols = 256;

    auto x = rand_vec(rows * cols, 5.f);
    auto w = rand_vec(cols, 2.f);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_fp32(x.data() + r * cols, expected.data() + r * cols,
                          w.data(), cols, 1e-5f);

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

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTest, FP32_FullHidden) {
    // 模拟真实模型维度: [B=2, S=128, hidden=3136]
    const size_t rows = 256, cols = 3136;

    auto x = rand_vec(rows * cols);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_fp32(x.data() + r * cols, expected.data() + r * cols,
                          w.data(), cols, 1e-5f);

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

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTest, FP32_GridStride) {
    // grid-stride loop: rows > gridDim.x 时验证正确性
    const size_t rows = 128, cols = 1024;
    const size_t grid = 4;

    auto x = rand_vec(rows * cols);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_fp32(x.data() + r * cols, expected.data() + r * cols,
                          w.data(), cols, 1e-5f);

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

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_w);
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

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_w);
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
        ref::rmsnorm_bf16(x.data() + r * cols, expected.data() + r * cols,
                          w.data(), cols, 1e-5f);

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

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_w);
}

TEST_F(RMSNormCorrectnessTestBF16, FullHidden) {
    const size_t rows = 32, cols = 3136;

    auto x = rand_vec(rows * cols, 2.f);
    auto w = rand_vec(cols);
    std::vector<float> expected(rows * cols), out(rows * cols);

    for (size_t r = 0; r < rows; ++r)
        ref::rmsnorm_bf16(x.data() + r * cols, expected.data() + r * cols,
                          w.data(), cols, 1e-5f);

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

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_w);
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
    printf("  [Perf] RMSNorm FP32:   %6.3f ms | %7.2f GB/s | %zu rows x %zu cols\n",
           ms, bw, (size_t)ROWS, (size_t)COLS);
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
    printf("  [Perf] RMSNorm BF16:   %6.3f ms | %7.2f GB/s | %zu rows x %zu cols\n",
           ms, bw, (size_t)ROWS, (size_t)COLS);
    EXPECT_GT(bw, 50.0);
}

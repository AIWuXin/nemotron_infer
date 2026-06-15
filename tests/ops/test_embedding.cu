// ===========================================================================
// test_embedding.cu — Embedding 查表测试（正确性 + 性能）
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/embedding.cuh"

using namespace nemotron;
using namespace nemotron::ops;

// ===========================================================================
// 1. CPU 参考
// ===========================================================================
namespace ref {

void embedding_lookup(const __nv_bfloat16* table,
                      const int64_t* token_ids,
                      __nv_bfloat16* out,
                      size_t num_tokens, size_t hidden) {
    for (size_t i = 0; i < num_tokens; ++i) {
        int64_t tok = token_ids[i];
        for (size_t j = 0; j < hidden; ++j)
            out[i * hidden + j] = table[tok * hidden + j];
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
// 3. 正确性测试
// ===========================================================================
class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
    }
};

TEST_F(EmbeddingTest, FP32_Small) {
    constexpr size_t vocab = 100, hidden = 128, num_tokens = 16;

    // Embedding table
    std::vector<__nv_bfloat16> table(vocab * hidden);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = __float2bfloat16_rn(float(i % 97) * 0.1f - 5.f);

    // Token IDs
    std::vector<int64_t> ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // CPU 参考
    std::vector<__nv_bfloat16> expected(num_tokens * hidden);
    ref::embedding_lookup(table.data(), ids.data(), expected.data(), num_tokens, hidden);

    // GPU
    auto d_table = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(vocab * hidden));
    auto d_ids   = allocate_tensor<int64_t>(TensorShape::make_1d(num_tokens));
    auto d_out   = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(num_tokens * hidden));
    copy_host_to_device(d_table, table.data());
    copy_host_to_device(d_ids, ids.data());
    cudaDeviceSynchronize();

    embedding_lookup<hidden>(d_table.data_, d_ids.data_, d_out.data_, num_tokens);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out(num_tokens * hidden);
    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < num_tokens * hidden; ++i)
        EXPECT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_table); free_tensor(d_ids); free_tensor(d_out);
}

TEST_F(EmbeddingTest, FullVocab) {
    // 用真实维度: vocab=131072, hidden=3136, 64 tokens
    constexpr size_t vocab = 131072, hidden = 3136, num_tokens = 64;

    std::vector<__nv_bfloat16> table(vocab * hidden);
    for (size_t i = 0; i < vocab * hidden; ++i)
        table[i] = __float2bfloat16_rn(float(i % 97) * 0.1f - 5.f);

    std::vector<int64_t> ids(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i)
        ids[i] = (i * 2047) % vocab;  // 分散访问，模拟真实分布

    std::vector<__nv_bfloat16> expected(num_tokens * hidden);
    ref::embedding_lookup(table.data(), ids.data(), expected.data(), num_tokens, hidden);

    auto d_table = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(vocab * hidden));
    auto d_ids   = allocate_tensor<int64_t>(TensorShape::make_1d(num_tokens));
    auto d_out   = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(num_tokens * hidden));
    copy_host_to_device(d_table, table.data());
    copy_host_to_device(d_ids, ids.data());
    cudaDeviceSynchronize();

    embedding_lookup<hidden>(d_table.data_, d_ids.data_, d_out.data_, num_tokens);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out(num_tokens * hidden);
    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < num_tokens * hidden; ++i)
        EXPECT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_table); free_tensor(d_ids); free_tensor(d_out);
}

TEST_F(EmbeddingTest, ScatteredAccess) {
    // 验证尾号 token 访问（接近 vocab 上限的 token）
    constexpr size_t vocab = 131072, hidden = 3136, num_tokens = 32;

    std::vector<__nv_bfloat16> table(vocab * hidden);
    for (size_t i = 0; i < vocab * hidden; ++i)
        table[i] = __float2bfloat16_rn(float(i) * 0.001f);

    // 访问前几个和后几个 token
    std::vector<int64_t> ids = {
        0, 1, 2, 3, 4, 5, 6, 7,
        vocab/2-4, vocab/2-3, vocab/2-2, vocab/2-1,
        vocab/2, vocab/2+1, vocab/2+2, vocab/2+3,
        vocab-8, vocab-7, vocab-6, vocab-5, vocab-4, vocab-3, vocab-2, vocab-1,
        // 补齐 32
        100, 200, 300, 400, 500, 600, 700, 800
    };

    std::vector<__nv_bfloat16> expected(num_tokens * hidden);
    ref::embedding_lookup(table.data(), ids.data(), expected.data(), num_tokens, hidden);

    auto d_table = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(vocab * hidden));
    auto d_ids   = allocate_tensor<int64_t>(TensorShape::make_1d(num_tokens));
    auto d_out   = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(num_tokens * hidden));
    copy_host_to_device(d_table, table.data());
    copy_host_to_device(d_ids, ids.data());
    cudaDeviceSynchronize();

    embedding_lookup<hidden>(d_table.data_, d_ids.data_, d_out.data_, num_tokens);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out(num_tokens * hidden);
    copy_device_to_host(out.data(), d_out);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < num_tokens * hidden; ++i)
        EXPECT_EQ(out[i], expected[i]) << " at index " << i;

    free_tensor(d_table); free_tensor(d_ids); free_tensor(d_out);
}

// ===========================================================================
// 4. 性能基准
// ===========================================================================
class EmbeddingPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();

        if (!d_table.data_) {
            d_table = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(VOCAB * HIDDEN));
            d_out   = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(NUM_TOKENS * HIDDEN));
            d_ids   = allocate_tensor<int64_t>(TensorShape::make_1d(NUM_TOKENS));
        }

        std::vector<__nv_bfloat16> h_table(VOCAB * HIDDEN);
        for (size_t i = 0; i < VOCAB * HIDDEN; ++i)
            h_table[i] = __float2bfloat16_rn(float(i % 97) * 0.1f);
        copy_host_to_device(d_table, h_table.data());

        std::vector<int64_t> h_ids(NUM_TOKENS);
        for (size_t i = 0; i < NUM_TOKENS; ++i)
            h_ids[i] = i * 2047 % VOCAB;
        copy_host_to_device(d_ids, h_ids.data());
        cudaDeviceSynchronize();
    }

    static constexpr size_t VOCAB = 131072;
    static constexpr size_t HIDDEN = 3136;
    static constexpr size_t NUM_TOKENS = 262144;  // 模拟 ~256K prefill

    Tensor<__nv_bfloat16> d_table, d_out;
    Tensor<int64_t> d_ids;
};

TEST_F(EmbeddingPerfTest, Bandwidth) {
    int warmup_iters = 5, bench_iters = 20;
    for (int i = 0; i < warmup_iters; ++i)
        embedding_lookup<HIDDEN>(d_table.data_, d_ids.data_, d_out.data_, NUM_TOKENS);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        embedding_lookup<HIDDEN>(d_table.data_, d_ids.data_, d_out.data_, NUM_TOKENS);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

    size_t bytes = NUM_TOKENS * HIDDEN * sizeof(__nv_bfloat16) * 2;  // read table + write out
    double bw = bytes / (ms * 1e6);
    printf("  [Perf] Embedding:       %6.3f ms | %7.2f GB/s | %zu tokens | %.1f MB read\n",
           ms, bw, (size_t)NUM_TOKENS, float(NUM_TOKENS * HIDDEN * sizeof(__nv_bfloat16)) / (1024*1024));
    EXPECT_GT(bw, 10.0);
}

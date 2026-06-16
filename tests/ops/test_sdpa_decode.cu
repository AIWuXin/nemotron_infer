// ===========================================================================
// test_sdpa_decode.cu — SDPA Decode 测试（单 token + cache 追加）
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/attention/sdpa_decode.cuh"

using namespace nemotron;
using namespace nemotron::ops::attention;

// ===========================================================================
// 1. CPU 参考
// ===========================================================================
namespace ref {

// 单步 decode: O = softmax(Q @ K[0..S-1]^T / sqrt(d)) @ V[0..S-1]
void sdpa_decode_fp32(const float* Q, const float* K, const float* V,
                      float* O, int S, int head_dim) {
    float rsqrt_d = 1.f / std::sqrt(float(head_dim));
    std::vector<float> score(S);
    for (int s = 0; s < S; ++s) {
        float sum = 0.f;
        for (int d = 0; d < head_dim; ++d)
            sum += Q[d] * K[s * head_dim + d];
        score[s] = sum * rsqrt_d;
    }

    float maxv = -INFINITY;
    for (int s = 0; s < S; ++s) maxv = std::max(maxv, score[s]);
    float sum = 0.f;
    for (int s = 0; s < S; ++s) { score[s] = std::exp(score[s] - maxv); sum += score[s]; }
    for (int s = 0; s < S; ++s) score[s] /= sum;

    for (int d = 0; d < head_dim; ++d) {
        float v = 0.f;
        for (int s = 0; s < S; ++s) v += score[s] * V[s * head_dim + d];
        O[d] = v;
    }
}

}  // namespace ref

// ===========================================================================
// 2. 辅助
// ===========================================================================
static void warmup_gpu() {
    float* buf = nullptr; cudaMalloc(&buf, 1024);
    cudaMemset(buf, 0, 1024); cudaDeviceSynchronize();
    cudaFree(buf);
}

static std::vector<float> rand_vec(size_t n, float r = 1.f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (float(rand()) / RAND_MAX * 2.f - 1.f) * r;
    return v;
}

// ===========================================================================
// 3. FP32 decode 正确性
// ===========================================================================
class SDPADecodeFP32Test : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); warmup_gpu(); }
};

TEST_F(SDPADecodeFP32Test, SingleStep) {
    const int H = 2, HEAD = 128, S = 32, TOTAL_S = 64;

    // 准备 cache 数据（K/V 各 S 个 token）
    auto K = rand_vec(H * S * HEAD);
    auto V = rand_vec(H * S * HEAD);
    auto Q = rand_vec(H * HEAD);
    std::vector<float> expected(H * HEAD), out(H * HEAD);

    for (int h = 0; h < H; ++h)
        ref::sdpa_decode_fp32(Q.data() + h * HEAD,
                              K.data() + h * S * HEAD,
                              V.data() + h * S * HEAD,
                              expected.data() + h * HEAD, S, HEAD);

    // GPU cache: 分配 TOTAL_S 容量，填入前 S 个
    auto d_K = allocate_tensor<float>(TensorShape::make_1d(H * TOTAL_S * HEAD));
    auto d_V = allocate_tensor<float>(TensorShape::make_1d(H * TOTAL_S * HEAD));
    auto d_O = allocate_tensor_zeros<float>(TensorShape::make_1d(H * HEAD));
    auto d_Q = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    auto d_Knew = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    auto d_Vnew = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));

    // 填充 cache 前 S 个位置（K 与 V 各自独立填充，cache layout [H, TOTAL_S, HEAD]）
    std::vector<float> initK(H * TOTAL_S * HEAD, 0.f), initV(H * TOTAL_S * HEAD, 0.f);
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < HEAD; ++d) {
                initK[h * TOTAL_S * HEAD + s * HEAD + d] = K[h * S * HEAD + s * HEAD + d];
                initV[h * TOTAL_S * HEAD + s * HEAD + d] = V[h * S * HEAD + s * HEAD + d];
            }
    copy_host_to_device(d_K, initK.data());
    copy_host_to_device(d_V, initV.data());
    copy_host_to_device(d_Q, Q.data());

    // K_new, V_new 填随机值（这次用不上，因为 S_cache=32 不包含它们）
    std::vector<float> Knew_h(H * HEAD), Vnew_h(H * HEAD);
    copy_host_to_device(d_Knew, Knew_h.data());
    copy_host_to_device(d_Vnew, Vnew_h.data());
    cudaDeviceSynchronize();

    sdpa_decode_fp32<HEAD>(d_Q.data_, d_K.data_, d_V.data_, d_O.data_,
                           d_Knew.data_, d_Vnew.data_, S, TOTAL_S, H);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_O);
    cudaDeviceSynchronize();

    for (int i = 0; i < H * HEAD; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-4f) << " at index " << i;

    // 验证 K_new 追加到了 position S
    std::vector<float> readback(H * TOTAL_S * HEAD);
    copy_device_to_host(readback.data(), d_K);
    cudaDeviceSynchronize();
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < HEAD; ++d) {
            float expected_k = Knew_h[h * HEAD + d];
            float got = readback[h * TOTAL_S * HEAD + S * HEAD + d];
            EXPECT_FLOAT_EQ(got, expected_k) << " K append h=" << h << " d=" << d;
        }

    free_tensor(d_Q); free_tensor(d_K); free_tensor(d_V);
    free_tensor(d_O); free_tensor(d_Knew); free_tensor(d_Vnew);
}

TEST_F(SDPADecodeFP32Test, MultiStep) {
    // 模拟 4 步 decode：每步追加新 KV，再算下一步
    const int H = 2, HEAD = 128, INIT_S = 8, TOTAL_S = 32, STEPS = 4;

    // 初始 cache
    auto K = rand_vec(H * TOTAL_S * HEAD);  // 实际用前 INIT_S 个
    auto V = rand_vec(H * TOTAL_S * HEAD);
    std::vector<float> expected(H * HEAD);

    // 预分配
    auto d_K = allocate_tensor<float>(TensorShape::make_1d(H * TOTAL_S * HEAD));
    auto d_V = allocate_tensor<float>(TensorShape::make_1d(H * TOTAL_S * HEAD));
    auto d_Q = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    auto d_O = allocate_tensor_zeros<float>(TensorShape::make_1d(H * HEAD));
    auto d_Knew = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    auto d_Vnew = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));

    // 初始 cache（K/V 各自独立填充前 INIT_S 个）
    std::vector<float> cacheK(H * TOTAL_S * HEAD, 0.f), cacheV(H * TOTAL_S * HEAD, 0.f);
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < INIT_S; ++s)
            for (int d = 0; d < HEAD; ++d) {
                cacheK[h * TOTAL_S * HEAD + s * HEAD + d] = K[h * TOTAL_S * HEAD + s * HEAD + d];
                cacheV[h * TOTAL_S * HEAD + s * HEAD + d] = V[h * TOTAL_S * HEAD + s * HEAD + d];
            }
    copy_host_to_device(d_K, cacheK.data());
    copy_host_to_device(d_V, cacheV.data());
    cudaDeviceSynchronize();

    int cur_S = INIT_S;
    for (int step = 0; step < STEPS; ++step) {
        auto Qh = rand_vec(H * HEAD);
        auto Kh = rand_vec(H * HEAD);
        auto Vh = rand_vec(H * HEAD);
        copy_host_to_device(d_Q, Qh.data());
        copy_host_to_device(d_Knew, Kh.data());
        copy_host_to_device(d_Vnew, Vh.data());

        // FP32 参考
        for (int h = 0; h < H; ++h) {
            // 从 K 中取前 cur_S 个 token 做 decode
            std::vector<float> K_slice(cur_S * HEAD), V_slice(cur_S * HEAD);
            int base = h * TOTAL_S * HEAD;
            for (int s = 0; s < cur_S; ++s)
                for (int d = 0; d < HEAD; ++d) {
                    K_slice[s * HEAD + d] = K[base + s * HEAD + d];
                    V_slice[s * HEAD + d] = V[base + s * HEAD + d];
                }
            ref::sdpa_decode_fp32(Qh.data() + h * HEAD,
                                  K_slice.data(), V_slice.data(),
                                  expected.data() + h * HEAD, cur_S, HEAD);
        }

        sdpa_decode_fp32<HEAD>(d_Q.data_, d_K.data_, d_V.data_, d_O.data_,
                               d_Knew.data_, d_Vnew.data_, cur_S, TOTAL_S, H);
        cudaDeviceSynchronize();

        std::vector<float> out(H * HEAD);
        copy_device_to_host(out.data(), d_O);
        cudaDeviceSynchronize();

        for (int i = 0; i < H * HEAD; ++i)
            EXPECT_NEAR(out[i], expected[i], 1e-4f) << " step=" << step << " idx=" << i;

        // 新 KV 写入 K, V（模拟 cache 更新）
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < HEAD; ++d) {
                K[h * TOTAL_S * HEAD + cur_S * HEAD + d] = Kh[h * HEAD + d];
                V[h * TOTAL_S * HEAD + cur_S * HEAD + d] = Vh[h * HEAD + d];
            }

        cur_S++;
    }

    free_tensor(d_Q); free_tensor(d_K); free_tensor(d_V);
    free_tensor(d_O); free_tensor(d_Knew); free_tensor(d_Vnew);
}

// GQA：Hq query heads 共享 Hkv KV heads；cache/K_new 按 kv_head 索引
TEST_F(SDPADecodeFP32Test, GQASingleStep) {
    const int Hq = 4, Hkv = 2, HEAD = 128, S = 32, TOTAL_S = 64;
    const int ratio = Hq / Hkv;

    auto Kkv = rand_vec(Hkv * S * HEAD);   // K/V cache 仅 Hkv 头
    auto Vkv = rand_vec(Hkv * S * HEAD);
    auto Q   = rand_vec(Hq * HEAD);
    std::vector<float> expected(Hq * HEAD), out(Hq * HEAD);

    for (int h = 0; h < Hq; ++h) {
        int kvh = h / ratio;
        ref::sdpa_decode_fp32(Q.data() + h * HEAD,
                              Kkv.data() + (size_t)kvh * S * HEAD,
                              Vkv.data() + (size_t)kvh * S * HEAD,
                              expected.data() + h * HEAD, S, HEAD);
    }

    auto d_K = allocate_tensor<float>(TensorShape::make_1d(Hkv * TOTAL_S * HEAD));
    auto d_V = allocate_tensor<float>(TensorShape::make_1d(Hkv * TOTAL_S * HEAD));
    auto d_O = allocate_tensor_zeros<float>(TensorShape::make_1d(Hq * HEAD));
    auto d_Q = allocate_tensor<float>(TensorShape::make_1d(Hq * HEAD));
    auto d_Kn = allocate_tensor<float>(TensorShape::make_1d(Hkv * HEAD));
    auto d_Vn = allocate_tensor<float>(TensorShape::make_1d(Hkv * HEAD));

    std::vector<float> initK(Hkv * TOTAL_S * HEAD, 0.f), initV(Hkv * TOTAL_S * HEAD, 0.f);
    for (int h = 0; h < Hkv; ++h)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < HEAD; ++d) {
                initK[h * TOTAL_S * HEAD + s * HEAD + d] = Kkv[h * S * HEAD + s * HEAD + d];
                initV[h * TOTAL_S * HEAD + s * HEAD + d] = Vkv[h * S * HEAD + s * HEAD + d];
            }
    std::vector<float> Knew_h(Hkv * HEAD), Vnew_h(Hkv * HEAD);
    for (auto& v : Knew_h) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Vnew_h) v = float(rand())/RAND_MAX-0.5f;
    copy_host_to_device(d_K, initK.data()); copy_host_to_device(d_V, initV.data());
    copy_host_to_device(d_Q, Q.data());
    copy_host_to_device(d_Kn, Knew_h.data()); copy_host_to_device(d_Vn, Vnew_h.data());
    cudaDeviceSynchronize();

    sdpa_decode_fp32<HEAD>(d_Q.data_, d_K.data_, d_V.data_, d_O.data_,
                           d_Kn.data_, d_Vn.data_, S, TOTAL_S, Hq, Hkv);
    cudaDeviceSynchronize();

    copy_device_to_host(out.data(), d_O); cudaDeviceSynchronize();
    for (int i = 0; i < Hq * HEAD; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-4f) << " at index " << i;

    // K_new 按 kv_head 追加到 position S（每个 kv 头一次）
    std::vector<float> readback(Hkv * TOTAL_S * HEAD);
    copy_device_to_host(readback.data(), d_K); cudaDeviceSynchronize();
    for (int h = 0; h < Hkv; ++h)
        for (int d = 0; d < HEAD; ++d)
            EXPECT_FLOAT_EQ(readback[h * TOTAL_S * HEAD + S * HEAD + d], Knew_h[h * HEAD + d])
                << " kv append h=" << h << " d=" << d;

    free_tensor(d_Q); free_tensor(d_K); free_tensor(d_V);
    free_tensor(d_O); free_tensor(d_Kn); free_tensor(d_Vn);
}

// ===========================================================================
// 4. BF16 正确性
// ===========================================================================
class SDPADecodeBF16Test : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); warmup_gpu(); }
};

TEST_F(SDPADecodeBF16Test, SingleStep) {
    const int H = 2, HEAD = 128, S = 64, TOTAL_S = 128;

    auto K_fp32 = rand_vec(H * S * HEAD);
    auto V_fp32 = rand_vec(H * S * HEAD);
    auto Q_fp32 = rand_vec(H * HEAD);
    std::vector<float> expected(H * HEAD);

    for (int h = 0; h < H; ++h)
        ref::sdpa_decode_fp32(Q_fp32.data() + h * HEAD,
                              K_fp32.data() + h * S * HEAD,
                              V_fp32.data() + h * S * HEAD,
                              expected.data() + h * HEAD, S, HEAD);

    // BF16
    std::vector<__nv_bfloat16> Q_bf16(H * HEAD), Knew_bf16(H * HEAD), Vnew_bf16(H * HEAD);
    std::vector<__nv_bfloat16> K_init(H * TOTAL_S * HEAD), V_init(H * TOTAL_S * HEAD);
    for (size_t i = 0; i < H * HEAD; ++i) {
        Q_bf16[i] = __float2bfloat16_rn(Q_fp32[i]);
        Knew_bf16[i] = __float2bfloat16_rn(Q_fp32[i]);  // dummy
        Vnew_bf16[i] = __float2bfloat16_rn(Q_fp32[i]);
    }
    for (size_t i = 0; i < H * TOTAL_S * HEAD; ++i) {
        K_init[i] = (i % (TOTAL_S * HEAD) < S * HEAD)
            ? __float2bfloat16_rn(K_fp32[i / (TOTAL_S * HEAD) * S * HEAD + i % (S * HEAD)])
            : __nv_bfloat16{};
    }

    auto d_K = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * TOTAL_S * HEAD));
    auto d_V = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * TOTAL_S * HEAD));
    auto d_Q = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
    auto d_O = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
    auto d_Kn = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
    auto d_Vn = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));

    // K/V cache 各自截前 S 个，用 K_fp32 / V_fp32 转 bf16 填（cache [H, TOTAL_S, HEAD]）
    std::vector<__nv_bfloat16> K_bf16(H * TOTAL_S * HEAD, __nv_bfloat16{});
    std::vector<__nv_bfloat16> V_bf16(H * TOTAL_S * HEAD, __nv_bfloat16{});
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < HEAD; ++d) {
                K_bf16[h * TOTAL_S * HEAD + s * HEAD + d] =
                    __float2bfloat16_rn(K_fp32[h * S * HEAD + s * HEAD + d]);
                V_bf16[h * TOTAL_S * HEAD + s * HEAD + d] =
                    __float2bfloat16_rn(V_fp32[h * S * HEAD + s * HEAD + d]);
            }

    copy_host_to_device(d_K, K_bf16.data());
    copy_host_to_device(d_V, V_bf16.data());
    copy_host_to_device(d_Q, Q_bf16.data());
    copy_host_to_device(d_Kn, Knew_bf16.data());
    copy_host_to_device(d_Vn, Vnew_bf16.data());
    cudaDeviceSynchronize();

    sdpa_decode_bf16<HEAD>(d_Q.data_, d_K.data_, d_V.data_, d_O.data_,
                           d_Kn.data_, d_Vn.data_, S, TOTAL_S, H);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out_bf16(H * HEAD);
    copy_device_to_host(out_bf16.data(), d_O);
    cudaDeviceSynchronize();

    for (int i = 0; i < H * HEAD; ++i)
        EXPECT_NEAR(__bfloat162float(out_bf16[i]), expected[i], 0.02f) << " at " << i;

    free_tensor(d_K); free_tensor(d_V); free_tensor(d_Q);
    free_tensor(d_O); free_tensor(d_Kn); free_tensor(d_Vn);
}

// ===========================================================================
// 5. 性能: decode 一次耗时（模拟累积到 256K）
// ===========================================================================
class SDPADecodePerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        default_allocator().reset();
        warmup_gpu();
        if (!fp32_K.data_) {
            fp32_K = allocate_tensor<float>(TensorShape::make_1d(H * TOTAL_S * HEAD));
            fp32_V = allocate_tensor<float>(TensorShape::make_1d(H * TOTAL_S * HEAD));
            fp32_O = allocate_tensor_zeros<float>(TensorShape::make_1d(H * HEAD));
        }
        if (!bf16_K.data_) {
            bf16_K = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * TOTAL_S * HEAD));
            bf16_V = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * TOTAL_S * HEAD));
            bf16_O = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
        }
    }

    static constexpr int H = 4;
    static constexpr int HEAD = 128;
    static constexpr int S = 262142;        // 当前 decode 时的 cache 长度
    static constexpr int TOTAL_S = 262144;

    Tensor<float> fp32_K, fp32_V, fp32_O;
    Tensor<__nv_bfloat16> bf16_K, bf16_V, bf16_O;
};

TEST_F(SDPADecodePerfTest, FP32) {
    auto Q = rand_vec(H * HEAD);
    auto Knew = rand_vec(H * HEAD);
    auto Vnew = rand_vec(H * HEAD);
    auto d_Q = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    auto d_Kn = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    auto d_Vn = allocate_tensor<float>(TensorShape::make_1d(H * HEAD));
    copy_host_to_device(d_Q, Q.data());
    copy_host_to_device(d_Kn, Knew.data());
    copy_host_to_device(d_Vn, Vnew.data());

    int warmup = 3, bench = 10;
    for (int i = 0; i < warmup; ++i)
        sdpa_decode_fp32<HEAD>(d_Q.data_, fp32_K.data_, fp32_V.data_, fp32_O.data_,
                               d_Kn.data_, d_Vn.data_, S, TOTAL_S, H);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        sdpa_decode_fp32<HEAD>(d_Q.data_, fp32_K.data_, fp32_V.data_, fp32_O.data_,
                               d_Kn.data_, d_Vn.data_, S, TOTAL_S, H);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench;
    printf("  [Perf] Decode FP32 S=%-5d     %7.3f ms\n", (int)S, ms);

    free_tensor(d_Q); free_tensor(d_Kn); free_tensor(d_Vn);
}

TEST_F(SDPADecodePerfTest, BF16) {
    auto Q = rand_vec(H * HEAD);
    auto Knew = rand_vec(H * HEAD);
    auto Vnew = rand_vec(H * HEAD);
    auto d_Q = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
    auto d_Kn = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
    auto d_Vn = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(H * HEAD));
    // 量化
    std::vector<__nv_bfloat16> Q_bf(H * HEAD), Kn_bf(H * HEAD), Vn_bf(H * HEAD);
    for (int i = 0; i < H * HEAD; ++i) {
        Q_bf[i] = __float2bfloat16_rn(Q[i]);
        Kn_bf[i] = __float2bfloat16_rn(Knew[i]);
        Vn_bf[i] = __float2bfloat16_rn(Vnew[i]);
    }
    copy_host_to_device(d_Q, Q_bf.data());
    copy_host_to_device(d_Kn, Kn_bf.data());
    copy_host_to_device(d_Vn, Vn_bf.data());

    int warmup = 3, bench = 10;
    for (int i = 0; i < warmup; ++i)
        sdpa_decode_bf16<HEAD>(d_Q.data_, bf16_K.data_, bf16_V.data_, bf16_O.data_,
                               d_Kn.data_, d_Vn.data_, S, TOTAL_S, H);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        sdpa_decode_bf16<HEAD>(d_Q.data_, bf16_K.data_, bf16_V.data_, bf16_O.data_,
                               d_Kn.data_, d_Vn.data_, S, TOTAL_S, H);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench;
    printf("  [Perf] Decode BF16 S=%-5d     %7.3f ms\n", (int)S, ms);

    free_tensor(d_Q); free_tensor(d_Kn); free_tensor(d_Vn);
}

// ===========================================================================
// test_ssd_scan.cu — Mamba-2 SSD scan prefill（递归形式，FP32）
// ===========================================================================
#include <gtest/gtest.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/mamba2/ssd_scan.cuh"

using namespace nemotron;
using namespace nemotron::ops::mamba2;

namespace ref {

inline float softplus_clamp(float v, float dt_min, float dt_max) {
    float sp = (v > 20.f) ? v : (v < -20.f) ? 0.f : std::log(1.f + std::exp(v));
    return std::min(dt_max, std::max(dt_min, sp));
}

// 顺序 SSM 递归 = SSD 的数学基准（等价 HF torch_forward 的 chunked 实现）
void ssd_scan_fp32(const float* x, const float* dt, const float* A_log,
                   const float* B, const float* C, const float* D_param,
                   const float* dt_bias, float* y, float* state_out,
                   int Bn, int S, int H, int D, int N, int G,
                   float dt_min, float dt_max) {
    std::vector<float> st(D * N);
    for (int b = 0; b < Bn; ++b) for (int h = 0; h < H; ++h) {
        const int group = h / (H / G);
        const float A_h = -std::exp(A_log[h]);
        const float Dh = D_param[h];
        const float bias = dt_bias ? dt_bias[h] : 0.f;
        std::fill(st.begin(), st.end(), 0.f);

        for (int t = 0; t < S; ++t) {
            const float dtv = softplus_clamp(dt[(b*S+t)*H+h] + bias, dt_min, dt_max);
            const float dA = std::exp(A_h * dtv);
            const float* Brow = &B[((b*S+t)*G+group)*N];
            const float* Crow = &C[((b*S+t)*G+group)*N];
            for (int p = 0; p < D; ++p) {
                const float xv = x[((b*S+t)*H+h)*D+p];
                float ydot = 0.f;
                float* sp = &st[p*N];
                for (int n = 0; n < N; ++n) {
                    sp[n] = sp[n] * dA + dtv * Brow[n] * xv;
                    ydot += sp[n] * Crow[n];
                }
                y[((b*S+t)*H+h)*D+p] = ydot + Dh * xv;
            }
        }
        if (state_out)
            for (int p = 0; p < D; ++p) for (int n = 0; n < N; ++n)
                state_out[((b*H+h)*D+p)*N+n] = st[p*N+n];
    }
}

}  // namespace ref

static void warmup() {
    float* b; cudaMalloc(&b, 1024); cudaMemset(b, 0, 1024); cudaDeviceSynchronize(); cudaFree(b);
}

class SSDScanTest : public ::testing::Test {
protected:
    void SetUp() override { default_allocator().reset(); warmup(); }
};

template<int H, int D, int N, int G>
static void run_case(int S, float tol) {
    const int B = 1;
    const float dt_min = 0.f, dt_max = FLT_MAX;   // 本模型 dt_limit=(0,inf)

    std::vector<float> x(B*S*H*D), dt(B*S*H), A_log(H), Bc(B*S*G*N), Cc(B*S*G*N),
                       Dc(H), dt_bias(H);
    for (auto& v : x)       v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log)   v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc)      v = float(rand())/RAND_MAX*0.1f;
    for (auto& v : dt_bias) v = float(rand())/RAND_MAX*0.2f;

    std::vector<float> y_ref(B*S*H*D), st_ref(B*H*D*N);
    ref::ssd_scan_fp32(x.data(), dt.data(), A_log.data(), Bc.data(), Cc.data(),
                       Dc.data(), dt_bias.data(), y_ref.data(), st_ref.data(),
                       B, S, H, D, N, G, dt_min, dt_max);

    auto d_x  = allocate_tensor<float>(TensorShape::make_1d(B*S*H*D));
    auto d_dt = allocate_tensor<float>(TensorShape::make_1d(B*S*H));
    auto d_A  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B  = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_C  = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_D  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_bias = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_y  = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*H*D));
    auto d_st = allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*D*N));
    copy_host_to_device(d_x, x.data());     copy_host_to_device(d_dt, dt.data());
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_B, Bc.data());
    copy_host_to_device(d_C, Cc.data());    copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_bias, dt_bias.data());
    cudaDeviceSynchronize();

    ssd_scan_prefill_fp32<H, D, N>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
        d_C.data_, d_D.data_, d_bias.data_, d_y.data_, d_st.data_,
        B, S, G, dt_min, dt_max);
    cudaDeviceSynchronize();

    std::vector<float> y_gpu(B*S*H*D), st_gpu(B*H*D*N);
    copy_device_to_host(y_gpu.data(), d_y);  copy_device_to_host(st_gpu.data(), d_st);
    cudaDeviceSynchronize();

    for (int i = 0; i < B*S*H*D; ++i)
        EXPECT_NEAR(y_gpu[i], y_ref[i], tol) << " y i=" << i;
    for (int i = 0; i < B*H*D*N; ++i)
        EXPECT_NEAR(st_gpu[i], st_ref[i], tol) << " state i=" << i;

    free_tensor(d_x); free_tensor(d_dt); free_tensor(d_A); free_tensor(d_B);
    free_tensor(d_C); free_tensor(d_D); free_tensor(d_bias); free_tensor(d_y); free_tensor(d_st);
}

TEST_F(SSDScanTest, FP32_Small)     { run_case<4, 8, 16, 2>(10, 1e-4f); }
TEST_F(SSDScanTest, FP32_FullMamba) { run_case<96, 80, 128, 8>(64, 1e-3f); }

// ===========================================================================
// 性能（单 chunk 长度 S=256）
// ===========================================================================
TEST_F(SSDScanTest, Perf) {
    const int H = 96, D = 80, N = 128, G = 8, S = 256, B = 1;
    auto d_x  = allocate_tensor<float>(TensorShape::make_1d(B*S*H*D));
    auto d_dt = allocate_tensor<float>(TensorShape::make_1d(B*S*H));
    auto d_A  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B  = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_C  = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_D  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_y  = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*H*D));

    int warm = 5, bench = 30;
    for (int i = 0; i < warm; ++i)
        ssd_scan_prefill_fp32<H, D, N>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
            d_C.data_, d_D.data_, nullptr, d_y.data_, nullptr, B, S, G);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        ssd_scan_prefill_fp32<H, D, N>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
            d_C.data_, d_D.data_, nullptr, d_y.data_, nullptr, B, S, G);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/bench;
    printf("  [Perf] SSD scan prefill    %6.3f ms  %d heads x %d dim x %d state x S=%d\n", ms, H, D, N, S);
    EXPECT_GT(ms, 0);
    free_tensor(d_x); free_tensor(d_dt); free_tensor(d_A); free_tensor(d_B);
    free_tensor(d_C); free_tensor(d_D); free_tensor(d_y);
}

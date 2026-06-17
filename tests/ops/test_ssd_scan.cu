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
#include "ops/mamba2/ssd_scan_chunked.cuh"
#include "ops/mamba2/ssd_scan_fused.cuh"

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

// bf16 IO 路径：x/dt/B/C/y 走 bf16，内部仍 fp32 累加；对 fp32 参考比对（bf16 容差）
template<int H, int D, int N, int G>
static void run_bf16_case(int S, float tol) {
    const int B = 1;
    const float dt_min = 0.f, dt_max = FLT_MAX;

    std::vector<float> x(B*S*H*D), dt(B*S*H), A_log(H), Bc(B*S*G*N), Cc(B*S*G*N),
                       Dc(H), dt_bias(H);
    for (auto& v : x)       v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log)   v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc)      v = float(rand())/RAND_MAX*0.1f;
    for (auto& v : dt_bias) v = float(rand())/RAND_MAX*0.2f;

    // 参考用 bf16-round 后的输入（与 GPU 同源），隔离 IO 量化误差，只验内核数学
    auto bf = [](float v){ return __bfloat162float(__float2bfloat16_rn(v)); };
    std::vector<float> xb=x, dtb=dt, Bb=Bc, Cb=Cc;
    for (auto& v : xb)  v = bf(v);
    for (auto& v : dtb) v = bf(v);
    for (auto& v : Bb)  v = bf(v);
    for (auto& v : Cb)  v = bf(v);

    std::vector<float> y_ref(B*S*H*D), st_ref(B*H*D*N);
    ref::ssd_scan_fp32(xb.data(), dtb.data(), A_log.data(), Bb.data(), Cb.data(),
                       Dc.data(), dt_bias.data(), y_ref.data(), st_ref.data(),
                       B, S, H, D, N, G, dt_min, dt_max);

    std::vector<__nv_bfloat16> xh(B*S*H*D), dth(B*S*H), Bh(B*S*G*N), Ch(B*S*G*N);
    for (size_t i=0;i<xh.size();++i) xh[i]=__float2bfloat16_rn(x[i]);
    for (size_t i=0;i<dth.size();++i) dth[i]=__float2bfloat16_rn(dt[i]);
    for (size_t i=0;i<Bh.size();++i){ Bh[i]=__float2bfloat16_rn(Bc[i]); Ch[i]=__float2bfloat16_rn(Cc[i]); }

    auto d_x  = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*S*H*D));
    auto d_dt = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*S*H));
    auto d_A  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B  = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*S*G*N));
    auto d_C  = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*S*G*N));
    auto d_D  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_bias = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_y  = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(B*S*H*D));
    copy_host_to_device(d_x, xh.data());  copy_host_to_device(d_dt, dth.data());
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_B, Bh.data());
    copy_host_to_device(d_C, Ch.data());  copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_bias, dt_bias.data());
    cudaDeviceSynchronize();

    ssd_scan_prefill_bf16<H, D, N>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
        d_C.data_, d_D.data_, d_bias.data_, d_y.data_, nullptr,
        B, S, G, dt_min, dt_max);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> y_gpu(B*S*H*D);
    copy_device_to_host(y_gpu.data(), d_y);
    cudaDeviceSynchronize();

    for (int i = 0; i < B*S*H*D; ++i)
        EXPECT_NEAR(__bfloat162float(y_gpu[i]), y_ref[i], tol) << " y i=" << i;
}

TEST_F(SSDScanTest, BF16_Small)     { run_bf16_case<4, 8, 16, 2>(10, 5e-2f); }
TEST_F(SSDScanTest, BF16_FullMamba) { run_bf16_case<96, 80, 128, 8>(64, 1.5e-1f); }

// ===========================================================================
// Chunked SSD（TF32）正确性：对顺序递归参考（= HF 等价）。chunk 数学对任意 C
// 都等价递归，故任意 C 必须匹配（容差取 TF32 量级）。
// ===========================================================================
template<int H, int P, int N, int G, int C>
static void run_chunked_case(int S, float tol) {
    const int B = 1;
    const float dt_min = 0.f, dt_max = FLT_MAX;

    std::vector<float> x(B*S*H*P), dt(B*S*H), A_log(H), Bc(B*S*G*N), Cc(B*S*G*N),
                       Dc(H), dt_bias(H);
    for (auto& v : x)       v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log)   v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc)      v = float(rand())/RAND_MAX*0.1f;
    for (auto& v : dt_bias) v = float(rand())/RAND_MAX*0.2f;

    std::vector<float> y_ref(B*S*H*P), st_ref(B*H*P*N);
    ref::ssd_scan_fp32(x.data(), dt.data(), A_log.data(), Bc.data(), Cc.data(),
                       Dc.data(), dt_bias.data(), y_ref.data(), st_ref.data(),
                       B, S, H, P, N, G, dt_min, dt_max);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*S*H*P));
    auto d_dt= allocate_tensor<float>(TensorShape::make_1d(B*S*H));
    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_C = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_bias = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*H*P));
    auto d_st= allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*P*N));
    copy_host_to_device(d_x, x.data());   copy_host_to_device(d_dt, dt.data());
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_B, Bc.data());
    copy_host_to_device(d_C, Cc.data());  copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_bias, dt_bias.data());
    cudaDeviceSynchronize();

    ssd_scan_chunked_prefill_fp32<H, P, N, C>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
        d_C.data_, d_D.data_, d_bias.data_, d_y.data_, d_st.data_, B, S, G, dt_min, dt_max);
    cudaDeviceSynchronize();

    std::vector<float> y_gpu(B*S*H*P), st_gpu(B*H*P*N);
    copy_device_to_host(y_gpu.data(), d_y);  copy_device_to_host(st_gpu.data(), d_st);
    cudaDeviceSynchronize();

    double max_err = 0;
    for (int i = 0; i < B*S*H*P; ++i) {
        max_err = std::max(max_err, (double)std::abs(y_gpu[i]-y_ref[i]));
        EXPECT_NEAR(y_gpu[i], y_ref[i], tol) << " y i=" << i;
    }
    for (int i = 0; i < B*H*P*N; ++i)
        EXPECT_NEAR(st_gpu[i], st_ref[i], tol) << " state i=" << i;
    printf("  [Chunked C=%d] S=%d max_err=%.3e\n", C, S, max_err);
}

TEST_F(SSDScanTest, Chunked_Small)     { run_chunked_case<4, 8, 16, 2, 4>(10, 2e-2f); }   // nc=3，含 inter-chunk
TEST_F(SSDScanTest, Chunked_Pad)       { run_chunked_case<4, 8, 16, 2, 8>(20, 2e-2f); }   // S 非 C 整数倍
TEST_F(SSDScanTest, Chunked_FullMamba) { run_chunked_case<96, 80, 128, 8, 64>(256, 6e-2f); }

// fused（flash）正确性：内核体内复用 run_chunked_case 的随机+参考，仅换 fused 调用
template<int H, int P, int N, int G, int C>
static void run_fused_case(int S, float tol) {
    const int B = 1;
    const float dt_min = 0.f, dt_max = FLT_MAX;
    std::vector<float> x(B*S*H*P), dt(B*S*H), A_log(H), Bc(B*S*G*N), Cc(B*S*G*N), Dc(H), dt_bias(H);
    for (auto& v : x)       v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log)   v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc)      v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc)      v = float(rand())/RAND_MAX*0.1f;
    for (auto& v : dt_bias) v = float(rand())/RAND_MAX*0.2f;

    std::vector<float> y_ref(B*S*H*P), st_ref(B*H*P*N);
    ref::ssd_scan_fp32(x.data(), dt.data(), A_log.data(), Bc.data(), Cc.data(),
                       Dc.data(), dt_bias.data(), y_ref.data(), st_ref.data(),
                       B, S, H, P, N, G, dt_min, dt_max);

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*S*H*P));
    auto d_dt= allocate_tensor<float>(TensorShape::make_1d(B*S*H));
    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_C = allocate_tensor<float>(TensorShape::make_1d(B*S*G*N));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_bias = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*S*H*P));
    auto d_st= allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*P*N));
    copy_host_to_device(d_x, x.data());   copy_host_to_device(d_dt, dt.data());
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_B, Bc.data());
    copy_host_to_device(d_C, Cc.data());  copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_bias, dt_bias.data());
    cudaDeviceSynchronize();

    ssd_scan_fused_prefill_fp32<H, P, N, C>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
        d_C.data_, d_D.data_, d_bias.data_, d_y.data_, d_st.data_, B, S, G, dt_min, dt_max);
    cudaDeviceSynchronize();

    std::vector<float> y_gpu(B*S*H*P), st_gpu(B*H*P*N);
    copy_device_to_host(y_gpu.data(), d_y);  copy_device_to_host(st_gpu.data(), d_st);
    cudaDeviceSynchronize();

    double max_err = 0;
    for (int i = 0; i < B*S*H*P; ++i) {
        max_err = std::max(max_err, (double)std::abs(y_gpu[i]-y_ref[i]));
        EXPECT_NEAR(y_gpu[i], y_ref[i], tol) << " y i=" << i;
    }
    for (int i = 0; i < B*H*P*N; ++i)
        EXPECT_NEAR(st_gpu[i], st_ref[i], tol) << " state i=" << i;
    printf("  [Fused C=%d] S=%d max_err=%.3e\n", C, S, max_err);
}

TEST_F(SSDScanTest, Fused_Small)     { run_fused_case<4, 8, 16, 2, 4>(10, 2e-2f); }
TEST_F(SSDScanTest, Fused_Pad)       { run_fused_case<4, 8, 16, 2, 8>(20, 2e-2f); }
TEST_F(SSDScanTest, Fused_FullMamba) { run_fused_case<96, 80, 128, 8, 32>(256, 6e-2f); }

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

// ===========================================================================
// 串行 vs chunked(TF32) 测速：扫 S 与 chunk 大小。每个变体独立 reset()+分配输入+
// mark，warm/bench 间 reset_to 复用 workspace（与 block sweep 同套路）。
// ===========================================================================
template<int H, int P, int N, int G, int MODE, int C>
static double bench_scan(int S, int warm, int bench) {
    const int B = 1;
    default_allocator().reset();
    auto d_x = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B*S*H*P));
    auto d_dt= allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B*S*H));
    auto d_A = allocate_tensor_zeros<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B*S*G*N));
    auto d_C = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B*S*G*N));
    auto d_D = allocate_tensor_zeros<float>(TensorShape::make_1d(H));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d((int64_t)B*S*H*P));
    cudaDeviceSynchronize();
    auto mk = default_allocator().mark();
    auto fn = [&] {
        if constexpr (MODE == 0)
            ssd_scan_prefill_fp32<H, P, N>(d_x.data_, d_dt.data_, d_A.data_,
                d_B.data_, d_C.data_, d_D.data_, nullptr, d_y.data_, nullptr, B, S, G);
        else if constexpr (MODE == 1)
            ssd_scan_chunked_prefill_fp32<H, P, N, C>(d_x.data_, d_dt.data_, d_A.data_,
                d_B.data_, d_C.data_, d_D.data_, nullptr, d_y.data_, nullptr, B, S, G);
        else
            ssd_scan_fused_prefill_fp32<H, P, N, C>(d_x.data_, d_dt.data_, d_A.data_,
                d_B.data_, d_C.data_, d_D.data_, nullptr, d_y.data_, nullptr, B, S, G);
    };
    for (int i = 0; i < warm; ++i) { fn(); cudaDeviceSynchronize(); default_allocator().reset_to(mk); }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float tot = 0.f;
    for (int i = 0; i < bench; ++i) {
        cudaEventRecord(e0); fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms;
        default_allocator().reset_to(mk);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)tot / bench;
}

TEST_F(SSDScanTest, Perf_ChunkedVsSerial) {
    constexpr int H = 96, P = 80, N = 128, G = 8;
    const int Ss[] = {512, 2048, 8192};
    const int warm = 3, bench = 8;
    printf("  [Perf SSD] H=%d P=%d N=%d G=%d  (warm=%d bench=%d)\n", H, P, N, G, warm, bench);
    printf("  %6s | %11s | %16s | %16s %16s\n", "S", "serial(ms)", "fused C32", "chunkC64(物化)", "chunkC128");
    for (int S : Ss) {
        double ser  = bench_scan<H, P, N, G, 0, 0  >(S, warm, bench);
        double fz   = bench_scan<H, P, N, G, 2, 32 >(S, warm, bench);
        double c64  = bench_scan<H, P, N, G, 1, 64 >(S, warm, bench);
        double c128 = bench_scan<H, P, N, G, 1, 128>(S, warm, bench);
        printf("  %6d | %11.3f | %9.3f(%4.1fx) | %9.3f(%4.1fx) %9.3f(%4.1fx)\n",
               S, ser, fz, ser/fz, c64, ser/c64, c128, ser/c128);
        EXPECT_GT(fz, 0.0);
    }
    default_allocator().reset();
}

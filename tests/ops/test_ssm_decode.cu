// ===========================================================================
// test_ssm_decode.cu — Mamba-2 SSM 单步 decode
// ===========================================================================
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"
#include "ops/mamba2/ssm.cuh"

using namespace nemotron;
using namespace nemotron::ops::mamba2;

namespace ref {

void ssm_step_fp32(float x, float dt, float A_log, float D,
                   const float* B_row, const float* C_row,
                   float* state, float& y, int N) {
    float s = (dt > 20.f) ? dt : (dt < -20.f) ? 0.f : std::log(1.f + std::exp(dt));
    dt = s;   // 本模型 dt_limit=(0,inf)：只 softplus 不截断（与 kernel 默认 (0,FLT_MAX) 一致）
    float A = -std::exp(A_log);          // A=-exp(A_log)，不预乘 dt
    float dA = std::exp(dt * A);          // dA=exp(dt·A)=exp(-exp(A_log)·dt)，线性于 dt（对齐 prefill/HF）

    float y_dot = 0.f;
    for (int n = 0; n < N; ++n) {
        float dB = dt * B_row[n];
        state[n] = state[n] * dA + dB * x;
        y_dot += state[n] * C_row[n];
    }
    y = y_dot + D * x;
}

}  // namespace ref

static void warmup() {
    float* b; cudaMalloc(&b, 1024); cudaMemset(b,0,1024); cudaDeviceSynchronize(); cudaFree(b);
}

TEST(SSMDecodeTest, FP32_Small) {
    const int H = 4, D = 8, N = 16, G = 2, B = 1;
    std::vector<float> x(B*H*D), dt(B*H), A_log(H), Bc(B*G*N), Dc(H), y_gpu(B*H*D);
    std::vector<float> dt_bias(H, 0.f);
    for (auto& v : x) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log) v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc) v = float(rand())/RAND_MAX*0.1f;

    // CPU
    std::vector<float> Cc_h(B*G*N), Dc_h(H);
    for (auto& v : Cc_h) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc_h) v = float(rand())/RAND_MAX*0.1f;

    std::vector<float> cpu_state(B*H*D*N, 0);
    std::vector<float> y_ref(B*H*D);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        int g = h / (H/G);
        for (int d = 0; d < D; ++d) {
            ref::ssm_step_fp32(x[b*H*D+h*D+d], dt[b*H+h], A_log[h], Dc_h[h],
                               &Bc[b*G*N+g*N], &Cc_h[b*G*N+g*N],
                               &cpu_state[b*H*D*N+h*D*N+d*N], y_ref[b*H*D+h*D+d], N);
        }
    }

    std::vector<float> state_init(B*H*D*N, 0);
    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*H*D));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*D));
    auto d_dt = allocate_tensor<float>(TensorShape::make_1d(B*H));
    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_C = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(B*H*D*N));
    copy_host_to_device(d_x, x.data()); copy_host_to_device(d_y, y_gpu.data());
    copy_host_to_device(d_dt, dt.data()); copy_host_to_device(d_A, A_log.data());
    copy_host_to_device(d_B, Bc.data()); copy_host_to_device(d_C, Cc_h.data());
    copy_host_to_device(d_D, Dc_h.data()); copy_host_to_device(d_state, state_init.data());
    cudaDeviceSynchronize();

    ssm_decode_fp32<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                                        d_C.data_, d_D.data_, d_state.data_, d_y.data_,
                                        nullptr, B, G);
    cudaDeviceSynchronize();

    copy_device_to_host(y_gpu.data(), d_y); cudaDeviceSynchronize();
    for (int i = 0; i < B*H*D; ++i)
        EXPECT_NEAR(y_gpu[i], y_ref[i], 1e-5f) << " i=" << i;

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_dt); free_tensor(d_A);
    free_tensor(d_B); free_tensor(d_C); free_tensor(d_D); free_tensor(d_state);
}

TEST(SSMDecodeTest, FP32_FullMamba) {
    const int H = 96, D = 80, N = 128, G = 8, B = 1;
    std::vector<float> x(B*H*D), dt(B*H), A_log(H), Bc(B*G*N), Cc(B*G*N), Dc(H, 0), y_gpu(B*H*D);
    std::vector<float> state_init(B*H*D*N, 0);
    for (auto& v : x) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log) v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc) v = float(rand())/RAND_MAX*0.1f;

    std::vector<float> y_ref(B*H*D);
    std::vector<float> cpu_state(B*H*D*N, 0);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        int g = h / (H/G);
        for (int d = 0; d < D; ++d) {
            ref::ssm_step_fp32(x[b*H*D+h*D+d], dt[b*H+h], A_log[h], Dc[h],
                               &Bc[b*G*N+g*N], &Cc[b*G*N+g*N],
                               &cpu_state[b*H*D*N+h*D*N+d*N], y_ref[b*H*D+h*D+d], N);
        }
    }

    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*H*D));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*D));
    auto d_dt = allocate_tensor<float>(TensorShape::make_1d(B*H));
    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_C = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(B*H*D*N));
    copy_host_to_device(d_x, x.data()); copy_host_to_device(d_dt, dt.data());
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_B, Bc.data());
    copy_host_to_device(d_C, Cc.data()); copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_state, state_init.data()); cudaDeviceSynchronize();

    ssm_decode_fp32<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                          d_C.data_, d_D.data_, d_state.data_, d_y.data_,
                          nullptr, B, G);
    cudaDeviceSynchronize();
    copy_device_to_host(y_gpu.data(), d_y); cudaDeviceSynchronize();

    for (int i = 0; i < B*H*D; ++i)
        EXPECT_NEAR(y_gpu[i], y_ref[i], 1e-4f) << " i=" << i;

    free_tensor(d_x); free_tensor(d_y); free_tensor(d_dt); free_tensor(d_A);
    free_tensor(d_B); free_tensor(d_C); free_tensor(d_D); free_tensor(d_state);
}

TEST(SSMDecodeTest, MultiStep) {
    const int H = 4, D = 8, N = 16, G = 2, B = 1, STEPS = 8;
    std::vector<float> A_log(H), Dc(H), Bc(B*G*N), Cc(B*G*N);
    for (auto& v : A_log) v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc) v = float(rand())/RAND_MAX-0.5f;

    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_C = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_B, Bc.data()); copy_host_to_device(d_C, Cc.data());

    std::vector<float> cpu_state(B*H*D*N, 0);
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(B*H*D*N));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*D));
    copy_host_to_device(d_state, cpu_state.data()); cudaDeviceSynchronize();

    for (int step = 0; step < STEPS; ++step) {
        std::vector<float> x_h(B*H*D), dt_h(B*H), y_gpu(B*H*D), y_ref(B*H*D);
        for (auto& v : x_h) v = float(rand())/RAND_MAX-0.5f;
        for (auto& v : dt_h) v = float(rand())/RAND_MAX-0.5f;

        auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*H*D));
        auto d_dt = allocate_tensor<float>(TensorShape::make_1d(B*H));
        copy_host_to_device(d_x, x_h.data()); copy_host_to_device(d_dt, dt_h.data());

        ssm_decode_fp32<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                              d_C.data_, d_D.data_, d_state.data_, d_y.data_,
                              nullptr, B, G);
        cudaDeviceSynchronize();

        copy_device_to_host(y_gpu.data(), d_y); cudaDeviceSynchronize();

        // CPU ref：cpu_state 独立演进（初值 0，与 GPU 同起点），不能从 d_state 回灌，
        // 否则等于让 CPU 多推进一步。末尾再单独比对 GPU/CPU state 一致性。
        for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
            int g = h / (H/G);
            for (int d = 0; d < D; ++d)
                ref::ssm_step_fp32(x_h[b*H*D+h*D+d], dt_h[b*H+h], A_log[h], Dc[h],
                                   &Bc[b*G*N+g*N], &Cc[b*G*N+g*N],
                                   &cpu_state[b*H*D*N+h*D*N+d*N], y_ref[b*H*D+h*D+d], N);
        }

        for (int i = 0; i < B*H*D; ++i)
            EXPECT_NEAR(y_gpu[i], y_ref[i], 1e-4f) << " step=" << step << " i=" << i;

        free_tensor(d_x); free_tensor(d_dt);
    }
    // 验证 GPU state 与 CPU 一致
    std::vector<float> gpu_state(B*H*D*N);
    copy_device_to_host(gpu_state.data(), d_state); cudaDeviceSynchronize();
    for (int i = 0; i < B*H*D*N; ++i)
        EXPECT_NEAR(gpu_state[i], cpu_state[i], 1e-5f) << " state i=" << i;

    free_tensor(d_A); free_tensor(d_D); free_tensor(d_B); free_tensor(d_C);
    free_tensor(d_state); free_tensor(d_y);
}

// ===========================================================================
// BF16 内核：x/dt/B/C/y 走 bf16，state/A_log/D 保持 fp32
// ===========================================================================
static void run_bf16_case(int H, int D, int N, int G, float tol) {
    const int B = 1;
    std::vector<float> x(B*H*D), dt(B*H), A_log(H), Bc(B*G*N), Cc(B*G*N), Dc(H);
    for (auto& v : x) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : dt) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : A_log) v = float(rand())/RAND_MAX-0.5f-1.f;
    for (auto& v : Bc) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Cc) v = float(rand())/RAND_MAX-0.5f;
    for (auto& v : Dc) v = float(rand())/RAND_MAX*0.1f;

    // 把 bf16-I/O 量化后的输入喂给 CPU 参考，隔离输入量化、只比 GPU 计算+输出量化
    auto q = [](std::vector<float>& f, std::vector<__nv_bfloat16>& bf){
        bf.resize(f.size());
        for (size_t i = 0; i < f.size(); ++i) { bf[i] = __float2bfloat16_rn(f[i]); f[i] = __bfloat162float(bf[i]); }
    };
    std::vector<__nv_bfloat16> x_bf, dt_bf, B_bf, C_bf;
    q(x, x_bf); q(dt, dt_bf); q(Bc, B_bf); q(Cc, C_bf);

    std::vector<float> cpu_state(B*H*D*N, 0), y_ref(B*H*D);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        int g = h / (H/G);
        for (int d = 0; d < D; ++d)
            ref::ssm_step_fp32(x[b*H*D+h*D+d], dt[b*H+h], A_log[h], Dc[h],
                               &Bc[b*G*N+g*N], &Cc[b*G*N+g*N],
                               &cpu_state[b*H*D*N+h*D*N+d*N], y_ref[b*H*D+h*D+d], N);
    }

    std::vector<float> state_init(B*H*D*N, 0);
    auto d_x  = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*H*D));
    auto d_dt = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*H));
    auto d_B  = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*G*N));
    auto d_C  = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*G*N));
    auto d_y  = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(B*H*D));
    auto d_A  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_D  = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(B*H*D*N));
    copy_host_to_device(d_x, x_bf.data());  copy_host_to_device(d_dt, dt_bf.data());
    copy_host_to_device(d_B, B_bf.data());  copy_host_to_device(d_C, C_bf.data());
    copy_host_to_device(d_A, A_log.data()); copy_host_to_device(d_D, Dc.data());
    copy_host_to_device(d_state, state_init.data()); cudaDeviceSynchronize();

    // 运行时维度 → 编译期实例化（仅测试用到的两组）
    if (H==4 && D==8 && N==16)
        ssm_decode_bf16<4,8,16><<<B*H,128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
            d_C.data_, d_D.data_, d_state.data_, d_y.data_, nullptr, B, G);
    else
        ssm_decode_bf16<96,80,128><<<B*H,128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
            d_C.data_, d_D.data_, d_state.data_, d_y.data_, nullptr, B, G);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> y_bf(B*H*D);
    copy_device_to_host(y_bf.data(), d_y); cudaDeviceSynchronize();
    for (int i = 0; i < B*H*D; ++i)
        EXPECT_NEAR(__bfloat162float(y_bf[i]), y_ref[i], tol) << " i=" << i;

    free_tensor(d_x); free_tensor(d_dt); free_tensor(d_B); free_tensor(d_C);
    free_tensor(d_y); free_tensor(d_A); free_tensor(d_D); free_tensor(d_state);
}

TEST(SSMDecodeTest, BF16_Small)     { run_bf16_case(4, 8, 16, 2, 2e-2f); }   // N<32 lane masking
TEST(SSMDecodeTest, BF16_FullMamba) { run_bf16_case(96, 80, 128, 8, 3e-2f); } // N=128 warp 归约

// Perf
TEST(SSMDecodeTest, Perf) {
    const int H = 96, D = 80, N = 128, G = 8, B = 1;
    auto d_x = allocate_tensor<float>(TensorShape::make_1d(B*H*D));
    auto d_y = allocate_tensor_zeros<float>(TensorShape::make_1d(B*H*D));
    auto d_dt = allocate_tensor<float>(TensorShape::make_1d(B*H));
    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_B = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_C = allocate_tensor<float>(TensorShape::make_1d(B*G*N));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(B*H*D*N));

    int warm = 20, bench = 100;
    for (int i = 0; i < warm; ++i)
        ssm_decode_fp32<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                              d_C.data_, d_D.data_, d_state.data_, d_y.data_,
                              nullptr, B, G);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        ssm_decode_fp32<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                              d_C.data_, d_D.data_, d_state.data_, d_y.data_,
                              nullptr, B, G);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/bench;
    printf("  [Perf] SSM decode fp32     %6.3f ms  %d heads x %d dim x %d state\n", ms, H, D, N);
    EXPECT_GT(ms, 0);
    free_tensor(d_x); free_tensor(d_y); free_tensor(d_dt); free_tensor(d_A);
    free_tensor(d_B); free_tensor(d_C); free_tensor(d_D); free_tensor(d_state);
}

TEST(SSMDecodeTest, BF16_Perf) {
    const int H = 96, D = 80, N = 128, G = 8, B = 1;
    auto d_x = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*H*D));
    auto d_y = allocate_tensor_zeros<__nv_bfloat16>(TensorShape::make_1d(B*H*D));
    auto d_dt = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*H));
    auto d_B = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*G*N));
    auto d_C = allocate_tensor<__nv_bfloat16>(TensorShape::make_1d(B*G*N));
    auto d_A = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_D = allocate_tensor<float>(TensorShape::make_1d(H));
    auto d_state = allocate_tensor<float>(TensorShape::make_1d(B*H*D*N));

    int warm = 20, bench = 100;
    for (int i = 0; i < warm; ++i)
        ssm_decode_bf16<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                              d_C.data_, d_D.data_, d_state.data_, d_y.data_, nullptr, B, G);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; ++i)
        ssm_decode_bf16<H,D,N><<<B*H, 128>>>(d_x.data_, d_dt.data_, d_A.data_, d_B.data_,
                              d_C.data_, d_D.data_, d_state.data_, d_y.data_, nullptr, B, G);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/bench;
    printf("  [Perf] SSM decode bf16     %6.3f ms  %d heads x %d dim x %d state\n", ms, H, D, N);
    EXPECT_GT(ms, 0);
    free_tensor(d_x); free_tensor(d_y); free_tensor(d_dt); free_tensor(d_A);
    free_tensor(d_B); free_tensor(d_C); free_tensor(d_D); free_tensor(d_state);
}

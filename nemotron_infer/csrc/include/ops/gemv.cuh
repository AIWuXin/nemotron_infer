//
// gemv.cuh — 自定义 FP8 GEMV (M=1 decode 快路)
// ===========================================================================
// decode 单 token 时 W[N,K] @ x[K] 是纯带宽受限：每 token 读 ~5.4GB 权重，
// 满 DRAM 带宽(272 GB/s)的理论上限 ≈ 20ms = 50 t/s。要逼近这个上限，
// 关键是让足够多的 load 同时在飞以隐藏 DRAM 延迟。
//
// 设计：
//   - warp-per-row：每个 warp 算一行输出，32 lane 沿 K 用 uint4(16B=16 fp8) 向量读，
//     consecutive lane 读 consecutive 16B → 完全 coalesced。
//   - 主循环 4× 展开：一次发 4 个 uint4 load 进寄存器再统一消费，打满 memory-level
//     parallelism（否则 load→立即累加 acc 形成依赖链，退化成延迟受限，只有 ~11% 带宽）。
//   - 不用 shared memory 暂存 x：x 仅 K 个元素且被所有行复用，常驻 L2，直接 __ldg 读即可；
//     省掉 smem 才能拉满 occupancy（K=INTER=12544 时 smem 会到 25KB 压垮 occupancy）。
//
// 精度：激活 x 保持 bf16（W8A16），不量化。gemv 带宽全在读 W，量化 x 不省带宽，
//   反而掉精度、多两个 amax/quant kernel。
//
// 权重布局与 gemm_fp8 一致：W[N,K] row-major e4m3 + w_scale[N] per-row。
//   y[n] = (Σ_k float(W[n,k]) * float(x[k])) * w_scale[n]
// ===========================================================================
#ifndef NEMOTRON_INFER_OPS_GEMV_CUH
#define NEMOTRON_INFER_OPS_GEMV_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace nemotron::ops {

// 一个 uint4(16 个 fp8 权重) 与 16 个 bf16 激活做点积
__device__ __forceinline__ float gemv_dot16(uint4 wpacked, const __nv_bfloat16* xp) {
    const unsigned int words[4] = {wpacked.x, wpacked.y, wpacked.z, wpacked.w};
    float s = 0.f;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const unsigned int wd = words[b];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_fp8_e4m3 w8;
            w8.__x = (unsigned char)((wd >> (j * 8)) & 0xFFu);
            s += static_cast<float>(w8) * __bfloat162float(__ldg(xp + b * 4 + j));
        }
    }
    return s;
}

template<int ROWS_PER_BLOCK>
__global__ void gemv_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,    // [K]
    const __nv_fp8_e4m3* __restrict__ W,    // [N, K] row-major
    const float* __restrict__ w_scale,      // [N]
    __nv_bfloat16* __restrict__ y,          // [N]
    int N, int K
) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= N) return;
    const int lane = threadIdx.x;
    const __nv_fp8_e4m3* wrow = W + (size_t)row * K;

    const int K16 = K & ~15;
    const int nvec = K16 >> 4;                       // 16B 块数
    const uint4* wv = reinterpret_cast<const uint4*>(wrow);

    float acc = 0.f;
    int v = lane;
    // 4× 展开：4 个 uint4 load 同时在飞，再统一消费
    for (; v + 96 < nvec; v += 128) {
        uint4 p0 = wv[v];
        uint4 p1 = wv[v + 32];
        uint4 p2 = wv[v + 64];
        uint4 p3 = wv[v + 96];
        acc += gemv_dot16(p0, x + (v)      * 16);
        acc += gemv_dot16(p1, x + (v + 32) * 16);
        acc += gemv_dot16(p2, x + (v + 64) * 16);
        acc += gemv_dot16(p3, x + (v + 96) * 16);
    }
    for (; v < nvec; v += 32)
        acc += gemv_dot16(wv[v], x + v * 16);
    // 尾部（K 非 16 倍数，本模型不触发）
    for (int k = K16 + lane; k < K; k += 32)
        acc += static_cast<float>(wrow[k]) * __bfloat162float(__ldg(x + k));

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
        y[row] = __float2bfloat16_rn(acc * w_scale[row]);
}

// host wrapper：drop-in 替代 decode 路径里的 quantize_activation_fp8 + gemm_fp8。
inline void gemv_fp8(
    const __nv_bfloat16* x,
    const __nv_fp8_e4m3* W,
    const float* w_scale,
    __nv_bfloat16* y,
    int N, int K,
    cudaStream_t stream = nullptr
) {
    constexpr int ROWS = 4;                          // 128 线程/block，无 smem，高 occupancy
    dim3 block(32, ROWS);
    dim3 grid((N + ROWS - 1) / ROWS);
    gemv_fp8_kernel<ROWS><<<grid, block, 0, stream>>>(x, W, w_scale, y, N, K);
}

}  // namespace nemotron::ops

#endif  // NEMOTRON_INFER_OPS_GEMV_CUH

//
// Created by Administrator on 2026/6/8.
//

#ifndef NEMOTRON_INFER_REDUCE_CUH
#define NEMOTRON_INFER_REDUCE_CUH


#include <cuda_bf16.h>


namespace nemotron::ops {
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_BIT_SIZE = 5;  // 2 ** 5 = 32
    constexpr auto WARP_BIT = 0x1f;

    __device__ __forceinline__ float warp_reduce_sum(float val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        return val;
    }

    __device__ __forceinline__ float block_reduce_sum(float val) {
        const uint32_t lane_idx = threadIdx.x & WARP_BIT;
        const uint32_t wid_idx = threadIdx.x >> WARP_BIT_SIZE;
        const uint32_t max_wid = blockDim.x >> WARP_BIT_SIZE;

        val = warp_reduce_sum(val);

        // 一个block最多32个warp
        __shared__ float shared_sum[32];
        if (lane_idx == 0) shared_sum[wid_idx] = val;
        __syncthreads();

        // block内规约
        val = threadIdx.x < max_wid ? shared_sum[threadIdx.x] : 0.f;
        if (wid_idx == 0) {
            val = warp_reduce_sum(val);
            shared_sum[0] = val;
        }
        __syncthreads();

        return shared_sum[0];
    }


    /**
     * RMSNorm 前向 kernel（FP32）
     * 每 block 处理一行，grid-stride loop 覆盖多行
     * @param x      输入 [rows, cols]
     * @param y      输出 [rows, cols]
     * @param w      weight [cols]
     * @param rows   行数（通常 = B * S）
     * @param eps    数值稳定性常数（1e-5）
     **/
    __device__ __forceinline__ void rmsnorm_kernel_fp32(
        const float* __restrict__ x,
        float* __restrict__ y,
        const float* __restrict__ w,
        const size_t rows,
        const size_t cols,
        const float eps = 1e-5f
    ) {
        // grid-stride loop: 行方向循环
        for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
            const float* x_row = x + row * cols;
            float* y_row = y + row * cols;

            // Step 1: 每个线程加载自己的元素，算局部 x^2
            float rms = 0.f;
            for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
                float val = x_row[col];
                rms += val * val;
            }

            // Step 2: block reduce 得到该行的 sum(x^2)
            rms = block_reduce_sum(rms);

            // Step 3: 计算 RMS
            rms = rsqrtf(rms / static_cast<float>(cols) + eps);

            // Step 4: 每个线程写自己的输出
            for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
                y_row[col] = x_row[col] * rms * w[col];
            }

            __syncthreads();  // 准备处理下一行
        }
    }


    // ===========================================================================
    // __global__ launch kernel
    // ===========================================================================
    // static：非模板 __global__，header 多 TU 包含时给 TU-local 链接（避免 nvlink multiple def）
    static __global__ void rmsnorm_launch_fp32(
        const float* __restrict__ x,
        float* __restrict__ y,
        const float* __restrict__ w,
        const size_t rows,
        const size_t cols,
        const float eps
    ) {
        rmsnorm_kernel_fp32(x, y, w, rows, cols, eps);
    }

    // ===========================================================================
    // __host__ 调用接口
    // ===========================================================================
    inline __host__ void rmsnorm_fp32(
        const float* in0,
        float* out0,
        const float* weight,
        const size_t rows,
        const size_t cols,
        const float eps = 1e-5f,
        cudaStream_t stream = nullptr
    ) {
        constexpr int block_dimx = 256;
        const int grid_dimx = min(rows, static_cast<size_t>(8192));  // GPU grid 上限

        rmsnorm_launch_fp32<<<grid_dimx, block_dimx, 0, stream>>>(
            in0, out0, weight, rows, cols, eps
        );
    }


    // ===========================================================================
    // BF16 版本
    // ===========================================================================
    // 向量化 BF16 RMSNorm kernel（float4 一次读 8 个 BF16，cols 需为 8 的倍数）
    // note: 同样不会处理尾数，需要内存16字节对齐
    __device__ __forceinline__ void rmsnorm_kernel_bf16(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        const float* __restrict__ w,
        const size_t rows,
        const size_t cols,
        const float eps = 1e-5f
    ) {
        const size_t vec_size = cols / 8;
        const auto x_vec = reinterpret_cast<const float4*>(x);
        auto y_vec = reinterpret_cast<float4*>(y);

        for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
            const float4* x_row = x_vec + row * vec_size;
            float4* y_row = y_vec + row * vec_size;

            // Step 1: 向量化 load + 拆包算 x²（8 个 BF16 一次 load）
            float sq_sum = 0.f;
            for (size_t col_v = threadIdx.x; col_v < vec_size; col_v += blockDim.x) {
                const float4 in_val = x_row[col_v];

                const auto& bf_x = reinterpret_cast<const __nv_bfloat162&>(in_val.x);
                const auto& bf_y = reinterpret_cast<const __nv_bfloat162&>(in_val.y);
                const auto& bf_z = reinterpret_cast<const __nv_bfloat162&>(in_val.z);
                const auto& bf_w = reinterpret_cast<const __nv_bfloat162&>(in_val.w);

                const float2 f_x = __bfloat1622float2(bf_x);
                const float2 f_y = __bfloat1622float2(bf_y);
                const float2 f_z = __bfloat1622float2(bf_z);
                const float2 f_w = __bfloat1622float2(bf_w);

                sq_sum += f_x.x * f_x.x + f_x.y * f_x.y
                        + f_y.x * f_y.x + f_y.y * f_y.y
                        + f_z.x * f_z.x + f_z.y * f_z.y
                        + f_w.x * f_w.x + f_w.y * f_w.y;
            }

            sq_sum = block_reduce_sum(sq_sum);
            float rms = rsqrtf(sq_sum / cols + eps);

            // Step 2: 向量化 load + scale + 打包 store
            for (size_t col_v = threadIdx.x; col_v < vec_size; col_v += blockDim.x) {
                const float4 in_val = x_row[col_v];

                const auto& bf_x = reinterpret_cast<const __nv_bfloat162&>(in_val.x);
                const auto& bf_y = reinterpret_cast<const __nv_bfloat162&>(in_val.y);
                const auto& bf_z = reinterpret_cast<const __nv_bfloat162&>(in_val.z);
                const auto& bf_w = reinterpret_cast<const __nv_bfloat162&>(in_val.w);

                float2 f_x = __bfloat1622float2(bf_x);
                float2 f_y = __bfloat1622float2(bf_y);
                float2 f_z = __bfloat1622float2(bf_z);
                float2 f_w = __bfloat1622float2(bf_w);

                const float* w_col = w + col_v * 8;
                float scale = rms;
                f_x.x *= scale * w_col[0];
                f_x.y *= scale * w_col[1];
                f_y.x *= scale * w_col[2];
                f_y.y *= scale * w_col[3];
                f_z.x *= scale * w_col[4];
                f_z.y *= scale * w_col[5];
                f_w.x *= scale * w_col[6];
                f_w.y *= scale * w_col[7];

                __nv_bfloat162 out_x = __float22bfloat162_rn(f_x);
                __nv_bfloat162 out_y = __float22bfloat162_rn(f_y);
                __nv_bfloat162 out_z = __float22bfloat162_rn(f_z);
                __nv_bfloat162 out_w = __float22bfloat162_rn(f_w);

                y_row[col_v] = make_float4(
                    reinterpret_cast<const float&>(out_x),
                    reinterpret_cast<const float&>(out_y),
                    reinterpret_cast<const float&>(out_z),
                    reinterpret_cast<const float&>(out_w)
                );
            }

            __syncthreads();
        }
    }

    static __global__ void rmsnorm_launch_bf16(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        const float* __restrict__ w,
        const size_t rows,
        const size_t cols,
        const float eps
    ) {
        rmsnorm_kernel_bf16(x, y, w, rows, cols, eps);
    }

    inline __host__ void rmsnorm_bf16(
        const __nv_bfloat16* in0,
        __nv_bfloat16* out0,
        const float* weight,
        const size_t rows,
        const size_t cols,
        const float eps = 1e-5f,
        cudaStream_t stream = nullptr
    ) {
        constexpr int block_dimx = 1024;
        const int grid_dimx = min(rows, static_cast<size_t>(8192));

        rmsnorm_launch_bf16<<<grid_dimx, block_dimx, 0, stream>>>(
            in0, out0, weight, rows, cols, eps
        );
    }


    /**
     * 分组 block 规约 — 一趟 __syncthreads 完成 Group 组归约
     * 结构与 block_reduce_sum 严格一致，仅把 float 换为 float[Group]
     */
    template<int Group>
    __device__ __forceinline__ void block_group_reduce_sum(float* vals) {
        const uint32_t lane = threadIdx.x & WARP_BIT;
        const uint32_t wid  = threadIdx.x >> WARP_BIT_SIZE;
        const uint32_t num_warps = blockDim.x >> WARP_BIT_SIZE;

        // Step 1: 各 warp 内 Group 组独立做 butterfly reduction
        #pragma unroll
        for (int g = 0; g < Group; ++g)
            vals[g] = warp_reduce_sum(vals[g]);

        // Step 2: 每个 warp 的 lane 0 写 partial sum 到 shared
        //         布局: s[wid][g] — 第一维是 warp idx，跟 block_reduce_sum 对齐
        __shared__ float s[32][Group];
        if (lane == 0) {
            #pragma unroll
            for (int g = 0; g < Group; ++g)
                s[wid][g] = vals[g];
        }
        __syncthreads();

        // Step 3: 所有线程 gather（对照 block_reduce_sum 的
        //         val = threadIdx.x < max_wid ? shared[threadIdx.x] : 0.f）
        //         这里把 Group 个值一起装进 group 维度
        #pragma unroll
        for (int g = 0; g < Group; ++g)
            vals[g] = (threadIdx.x < num_warps) ? s[threadIdx.x][g] : 0.f;

        // Step 4: 只有 warp 0 做最终归约并写回（跟 block_reduce_sum 完全一致）
        if (wid == 0) {
            #pragma unroll
            for (int g = 0; g < Group; ++g)
                vals[g] = warp_reduce_sum(vals[g]);
            #pragma unroll
            for (int g = 0; g < Group; ++g)
                s[0][g] = vals[g];
        }
        __syncthreads();

        // Step 5: 所有线程读取最终结果（跟 block_reduce_sum return shared_sum[0] 一致）
        #pragma unroll
        for (int g = 0; g < Group; ++g)
            vals[g] = s[0][g];
    }

    // SiLU 标量：silu(z) = z * sigmoid(z) = z / (1+exp(-z))
    __device__ __forceinline__ float silu_scalar(float z) { return z / (1.f + expf(-z)); }

    // 门控 RMSNorm — 对齐 HF MambaRMSNormGated(norm_before_gate=False)：
    //   gated = x * silu(gate)
    //   rstd[g] = rsqrt(mean_over_group(gated^2) + eps)
    //   y = gated * rstd[group] * w
    // 注意：RMS 作用在 x*silu(gate) 上、gate 先过 SiLU（不是对 x 求 RMS、不是乘原值 gate）。
    template<int Group>
    __device__ __forceinline__ void rmsnorm_gated_kernel_fp32(
        const float* __restrict__ x,
        float* __restrict__ y,
        const float* __restrict__ w,
        const float* __restrict__ gate,
        const size_t rows,
        const size_t cols,
        const size_t group_size,
        const float eps = 1e-5f
    ) {
        for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
            const float* x_row = x + row * cols;
            const float* g_row = gate + row * cols;
            float* y_row = y + row * cols;

            // Step 1: 各 Group 求 gated=x*silu(gate) 的平方和
            float sq_sum[Group] = {0.f};
            for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
                float gated = x_row[col] * silu_scalar(g_row[col]);
                sq_sum[col / group_size] += gated * gated;
            }

            // Step 2: block 级组规约
            block_group_reduce_sum<Group>(sq_sum);

            // Step 3: 每组算 scale
            float scale[Group];
            #pragma unroll
            for (int g = 0; g < Group; ++g)
                scale[g] = rsqrtf(sq_sum[g] / static_cast<float>(group_size) + eps);

            // Step 4: 写输出: y = gated * rstd * w
            for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
                float gated = x_row[col] * silu_scalar(g_row[col]);
                y_row[col] = gated * scale[col / group_size] * w[col];
            }

            __syncthreads();
        }
    }

    template<int Group>
    __global__ void rmsnorm_gated_launch_fp32(
        const float* __restrict__ x,
        float* __restrict__ y,
        const float* __restrict__ w,
        const float* __restrict__ gate,
        const size_t rows,
        const size_t cols,
        const size_t group_size,
        const float eps
    ) {
        rmsnorm_gated_kernel_fp32<Group>(x, y, w, gate, rows, cols, group_size, eps);
    }

    template<int Group>
    inline __host__ void rmsnorm_gated_fp32(
        const float* in0,
        float* out0,
        const float* weight,
        const float* gate,
        const size_t rows,
        const size_t cols,
        const size_t group_size,
        const float eps = 1e-5f,
        cudaStream_t stream = nullptr
    ) {
        constexpr int block_dimx = 256;
        const int grid_dimx = min(rows, static_cast<size_t>(8192));
        rmsnorm_gated_launch_fp32<Group><<<grid_dimx, block_dimx, 0, stream>>>(
            in0, out0, weight, gate, rows, cols, group_size, eps
        );
    }


    // ===========================================================================
    // BF16 门控 RMSNorm（向量化）
    // ===========================================================================
    template<int Group>
    __device__ __forceinline__ void rmsnorm_gated_kernel_bf16(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        const float* __restrict__ w,
        const __nv_bfloat16* __restrict__ gate,
        const size_t rows,
        const size_t cols,
        const size_t group_size,
        const float eps = 1e-5f
    ) {
        const size_t vec_size = cols / 8;
        const auto x_vec = reinterpret_cast<const float4*>(x);
        const auto g_vec = reinterpret_cast<const float4*>(gate);
        auto y_vec = reinterpret_cast<float4*>(y);

        for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
            const float4* x_row = x_vec + row * vec_size;
            const float4* g_row = g_vec + row * vec_size;
            float4* y_row = y_vec + row * vec_size;

            // Step 1: 向量化 load x + gate，算 gated=x*silu(gate) 的平方和
            float sq_sum[Group] = {0.f};
            for (size_t col_v = threadIdx.x; col_v < vec_size; col_v += blockDim.x) {
                const float4 in_val = x_row[col_v];
                const float4 gate_val = g_row[col_v];
                float2 f_x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(in_val.x));
                float2 f_y = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(in_val.y));
                float2 f_z = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(in_val.z));
                float2 f_w = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(in_val.w));
                float2 gf_x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(gate_val.x));
                float2 gf_y = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(gate_val.y));
                float2 gf_z = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(gate_val.z));
                float2 gf_w = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(gate_val.w));
                f_x.x *= silu_scalar(gf_x.x); f_x.y *= silu_scalar(gf_x.y);
                f_y.x *= silu_scalar(gf_y.x); f_y.y *= silu_scalar(gf_y.y);
                f_z.x *= silu_scalar(gf_z.x); f_z.y *= silu_scalar(gf_z.y);
                f_w.x *= silu_scalar(gf_w.x); f_w.y *= silu_scalar(gf_w.y);

                size_t base = col_v * 8;
                sq_sum[base / group_size] += f_x.x * f_x.x;
                sq_sum[(base + 1) / group_size] += f_x.y * f_x.y;
                sq_sum[(base + 2) / group_size] += f_y.x * f_y.x;
                sq_sum[(base + 3) / group_size] += f_y.y * f_y.y;
                sq_sum[(base + 4) / group_size] += f_z.x * f_z.x;
                sq_sum[(base + 5) / group_size] += f_z.y * f_z.y;
                sq_sum[(base + 6) / group_size] += f_w.x * f_w.x;
                sq_sum[(base + 7) / group_size] += f_w.y * f_w.y;
            }

            block_group_reduce_sum<Group>(sq_sum);

            float scale[Group];
            #pragma unroll
            for (int g = 0; g < Group; ++g)
                scale[g] = rsqrtf(sq_sum[g] / static_cast<float>(group_size) + eps);

            // Step 4: load x + gate，scale，打包 store
            for (size_t col_v = threadIdx.x; col_v < vec_size; col_v += blockDim.x) {
                const float4 in_val = x_row[col_v];
                const float4 gate_val = g_row[col_v];

                const auto& bf_x = reinterpret_cast<const __nv_bfloat162&>(in_val.x);
                const auto& bf_y = reinterpret_cast<const __nv_bfloat162&>(in_val.y);
                const auto& bf_z = reinterpret_cast<const __nv_bfloat162&>(in_val.z);
                const auto& bf_w = reinterpret_cast<const __nv_bfloat162&>(in_val.w);

                const auto& g_bf_x = reinterpret_cast<const __nv_bfloat162&>(gate_val.x);
                const auto& g_bf_y = reinterpret_cast<const __nv_bfloat162&>(gate_val.y);
                const auto& g_bf_z = reinterpret_cast<const __nv_bfloat162&>(gate_val.z);
                const auto& g_bf_w = reinterpret_cast<const __nv_bfloat162&>(gate_val.w);

                float2 f_x = __bfloat1622float2(bf_x);
                float2 f_y = __bfloat1622float2(bf_y);
                float2 f_z = __bfloat1622float2(bf_z);
                float2 f_w = __bfloat1622float2(bf_w);

                float2 gf_x = __bfloat1622float2(g_bf_x);
                float2 gf_y = __bfloat1622float2(g_bf_y);
                float2 gf_z = __bfloat1622float2(g_bf_z);
                float2 gf_w = __bfloat1622float2(g_bf_w);

                const float* w_col = w + col_v * 8;
                size_t base = col_v * 8;
                // gated = x*silu(gate)，再 *rstd*w（gate 已含 SiLU，不再乘原值）
                f_x.x = f_x.x * silu_scalar(gf_x.x) * scale[base / group_size] * w_col[0];
                f_x.y = f_x.y * silu_scalar(gf_x.y) * scale[(base + 1) / group_size] * w_col[1];
                f_y.x = f_y.x * silu_scalar(gf_y.x) * scale[(base + 2) / group_size] * w_col[2];
                f_y.y = f_y.y * silu_scalar(gf_y.y) * scale[(base + 3) / group_size] * w_col[3];
                f_z.x = f_z.x * silu_scalar(gf_z.x) * scale[(base + 4) / group_size] * w_col[4];
                f_z.y = f_z.y * silu_scalar(gf_z.y) * scale[(base + 5) / group_size] * w_col[5];
                f_w.x = f_w.x * silu_scalar(gf_w.x) * scale[(base + 6) / group_size] * w_col[6];
                f_w.y = f_w.y * silu_scalar(gf_w.y) * scale[(base + 7) / group_size] * w_col[7];

                __nv_bfloat162 out_x = __float22bfloat162_rn(f_x);
                __nv_bfloat162 out_y = __float22bfloat162_rn(f_y);
                __nv_bfloat162 out_z = __float22bfloat162_rn(f_z);
                __nv_bfloat162 out_w = __float22bfloat162_rn(f_w);

                y_row[col_v] = make_float4(
                    reinterpret_cast<const float&>(out_x),
                    reinterpret_cast<const float&>(out_y),
                    reinterpret_cast<const float&>(out_z),
                    reinterpret_cast<const float&>(out_w)
                );
            }

            __syncthreads();
        }
    }


    template<int Group>
    __global__ void rmsnorm_gated_launch_bf16(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        const float* __restrict__ w,
        const __nv_bfloat16* __restrict__ gate,
        const size_t rows,
        const size_t cols,
        const size_t group_size,
        const float eps
    ) {
        rmsnorm_gated_kernel_bf16<Group>(x, y, w, gate, rows, cols, group_size, eps);
    }


    template<int Group>
    __host__ void rmsnorm_gated_bf16(
        const __nv_bfloat16* in0,
        __nv_bfloat16* out0,
        const float* weight,
        const __nv_bfloat16* gate,
        const size_t rows,
        const size_t cols,
        const size_t group_size,
        const float eps = 1e-5f,
        cudaStream_t stream = nullptr
    ) {
        constexpr int block_dimx = 256;
        const int grid_dimx = min(rows, static_cast<size_t>(8192));
        rmsnorm_gated_launch_bf16<Group><<<grid_dimx, block_dimx, 0, stream>>>(
            in0, out0, weight, gate, rows, cols, group_size, eps
        );
    }


    // ===========================================================================
    // Segmented Cumsum — 下三角衰减矩阵
    // ===========================================================================

    /**
     * 输入已经算好的 cumsum A_cum [L]，输出下三角矩阵
     * L[t][s] = exp(A_cum[t] - A_cum[s])  s ≤ t
     *            0                          s > t
     *
     * 每 block 处理一个 (b, h, c, t) 的行，grid = B * H * C * L
     * block 内 threads 对齐 L
     */
    template<int L>
    __device__ __forceinline__ void segment_sum_kernel_fp32(
        const float* __restrict__ A_cum,      // [num_triplets, L]
        float* __restrict__ L_out,            // [num_triplets, L, L]
        const size_t num_triplets             // B * H * C
    ) {
        // 每个 block 处理一行 t，多个 triplets 通过 grid-stride loop 覆盖
        for (size_t idx = blockIdx.x; idx < num_triplets * L; idx += gridDim.x) {
            const size_t triplet = idx / L;   // 哪个 (b,h,c)
            const size_t t = idx % L;         // 行号

            const float* A = A_cum + triplet * L;
            float* L_row = L_out + triplet * L * L + t * L;

            const float A_t = A[t];           // 所有列共享这个值

            // 每个线程算自己那列的 L[t][s]
            for (size_t s = threadIdx.x; s < L; s += blockDim.x) {
                L_row[s] = (s <= t) ? expf(A_t - A[s]) : 0.f;
            }
        }
    }

    template<int L>
    __global__ void segment_sum_launch_fp32(
        const float* __restrict__ A_cum,
        float* __restrict__ L_out,
        const size_t num_triplets
    ) {
        segment_sum_kernel_fp32<L>(A_cum, L_out, num_triplets);
    }

    template<int L>
    __host__ void segment_sum_fp32(
        const float* A_cum,
        float* L_out,
        const size_t num_triplets,
        cudaStream_t stream = nullptr
    ) {
        constexpr int block_dimx = L;         // 一行一个 block，线程数 = chunk_size
        const int grid_dimx = min((size_t)(num_triplets * L), (size_t)16384);
        segment_sum_launch_fp32<L><<<grid_dimx, block_dimx, 0, stream>>>(
            A_cum, L_out, num_triplets
        );
    }


    // ===========================================================================
    // BF16 版本 — float4 向量化 load/store（一次处理 8 个 BF16），内部 FP32 计算
    // ===========================================================================
    template<int L>
    __device__ __forceinline__ void segment_sum_kernel_bf16(
        const __nv_bfloat16* __restrict__ A_cum,
        __nv_bfloat16* __restrict__ L_out,
        const size_t num_triplets
    ) {
        constexpr int VEC = 8;
        const auto A_vec = reinterpret_cast<const float4*>(A_cum);

        for (size_t idx = blockIdx.x; idx < num_triplets * L; idx += gridDim.x) {
            const size_t triplet = idx / L;
            const size_t t = idx % L;

            const float4* A4 = A_vec + triplet * (L / VEC);
            auto* L_row = reinterpret_cast<float4*>(L_out + triplet * L * L + t * L);

            // ① 标量读 A[t]
            const int t_vec = t / VEC;
            const int t_off = t % VEC;
            const float A_t = __bfloat162float(
                reinterpret_cast<const __nv_bfloat16*>(A4 + t_vec)[t_off]
            );

            // ② 向量化 load 8 个 A[s]（一次 float4）
            const size_t s_vec = threadIdx.x;
            const float4 a_val = A4[s_vec];
            const auto& bf_x = reinterpret_cast<const __nv_bfloat162&>(a_val.x);
            const auto& bf_y = reinterpret_cast<const __nv_bfloat162&>(a_val.y);
            const auto& bf_z = reinterpret_cast<const __nv_bfloat162&>(a_val.z);
            const auto& bf_w = reinterpret_cast<const __nv_bfloat162&>(a_val.w);
            float2 f_x = __bfloat1622float2(bf_x);
            float2 f_y = __bfloat1622float2(bf_y);
            float2 f_z = __bfloat1622float2(bf_z);
            float2 f_w = __bfloat1622float2(bf_w);

            // ③ FP32 计算，逐列判断下三角
            size_t s_base = s_vec * VEC;
            if (s_base <= t) f_x.x = expf(A_t - f_x.x); else f_x.x = 0.f; s_base++;
            if (s_base <= t) f_x.y = expf(A_t - f_x.y); else f_x.y = 0.f; s_base++;
            if (s_base <= t) f_y.x = expf(A_t - f_y.x); else f_y.x = 0.f; s_base++;
            if (s_base <= t) f_y.y = expf(A_t - f_y.y); else f_y.y = 0.f; s_base++;
            if (s_base <= t) f_z.x = expf(A_t - f_z.x); else f_z.x = 0.f; s_base++;
            if (s_base <= t) f_z.y = expf(A_t - f_z.y); else f_z.y = 0.f; s_base++;
            if (s_base <= t) f_w.x = expf(A_t - f_w.x); else f_w.x = 0.f; s_base++;
            if (s_base <= t) f_w.y = expf(A_t - f_w.y); else f_w.y = 0.f;

            // ④ 向量化 store 8 个结果（一次 float4）
            __nv_bfloat162 out_x = __float22bfloat162_rn(f_x);
            __nv_bfloat162 out_y = __float22bfloat162_rn(f_y);
            __nv_bfloat162 out_z = __float22bfloat162_rn(f_z);
            __nv_bfloat162 out_w = __float22bfloat162_rn(f_w);

            L_row[s_vec] = make_float4(
                reinterpret_cast<const float&>(out_x),
                reinterpret_cast<const float&>(out_y),
                reinterpret_cast<const float&>(out_z),
                reinterpret_cast<const float&>(out_w)
            );
        }
    }

    template<int L>
    __global__ void segment_sum_launch_bf16(
        const __nv_bfloat16* __restrict__ A_cum,
        __nv_bfloat16* __restrict__ L_out,
        const size_t num_triplets
    ) {
        segment_sum_kernel_bf16<L>(A_cum, L_out, num_triplets);
    }

    template<int L>
    __host__ void segment_sum_bf16(
        const __nv_bfloat16* A_cum,
        __nv_bfloat16* L_out,
        const size_t num_triplets,
        cudaStream_t stream = nullptr
    ) {
        constexpr int block_dimx = L / 8;
        const int grid_dimx = min(num_triplets * L, static_cast<size_t>(65535));
        segment_sum_launch_bf16<L><<<grid_dimx, block_dimx, 0, stream>>>(
            A_cum, L_out, num_triplets
        );
    }
}


#endif //NEMOTRON_INFER_REDUCE_CUH

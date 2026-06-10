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
    __global__ void rmsnorm_launch_fp32(
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

    __global__ void rmsnorm_launch_bf16(
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
     * 分组block规约
     **/
    template<int Group>
    __device__ __forceinline__ void block_group_reduce_sum(float* vals) {
        const uint32_t lane_idx = threadIdx.x & WARP_BIT;
        const uint32_t wid_idx = threadIdx.x >> WARP_BIT_SIZE;
        const uint32_t max_wid = blockDim.x >> WARP_BIT_SIZE;

        #pragma unroll
        for (int group_idx = 0; group_idx < Group; ++group_idx) {
            vals[group_idx] = warp_reduce_sum(vals[group_idx]);
        }

        __shared__ float shared[Group][32];
        if (lane_idx == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < Group; ++group_idx) {
                shared[group_idx][wid_idx] = vals[group_idx];
            }
        }
        __syncthreads();

        if (wid_idx == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < Group; ++group_idx) {
                vals[group_idx] = threadIdx.x < max_wid ? shared[group_idx][lane_idx] : 0.f;
                vals[group_idx] = warp_reduce_sum(vals[group_idx]);
            }
        }
        __syncthreads();

        if (threadIdx.x < Group) shared[threadIdx.x][0] = vals[threadIdx.x];
        __syncthreads();
        #pragma unroll
        for (int group_idx = 0; group_idx < Group; ++group_idx) vals[group_idx] = shared[group_idx][0];
    }

    template<int Group>
    __device__ __forceinline__ void rmsnorm_gated_kernel_fp32(
        const float* __restrict__ x,
        float* __restrict__ y,
        const float* __restrict__ w,
        const float* __restrict__ gate,
        const float* __restrict__ 
        const size_t rows,
        const size_t cols,
        const float eps = 1e-5f
    ) {

    }
}


#endif //NEMOTRON_INFER_REDUCE_CUH

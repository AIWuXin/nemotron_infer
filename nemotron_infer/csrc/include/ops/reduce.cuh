//
// Created by Administrator on 2026/6/8.
//

#ifndef NEMOTRON_INFER_REDUCE_CUH
#define NEMOTRON_INFER_REDUCE_CUH


#include <cuda_bf16.h>


namespace nemotron::ops {
    constexpr int WARP_SIZE = 32;

    __device__ __forceinline__ float warp_reduce_sum(float val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        return val;
    }

    __device__ __forceinline__ float block_reduce_sum(float val) {
        const int lane_idx = threadIdx.x % WARP_SIZE;
        const int wid_idx = threadIdx.x / WARP_SIZE;
        const int max_wid = blockDim.x / WARP_SIZE;

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
            rms = rsqrtf(rms / cols + eps);

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
        const int grid_dimx = min(rows, (size_t)65535);  // GPU grid 上限

        rmsnorm_launch_fp32<<<grid_dimx, block_dimx, 0, stream>>>(
            in0, out0, weight, rows, cols, eps
        );
    }


    // ===========================================================================
    // BF16 版本
    // ===========================================================================

    __device__ __forceinline__ void rmsnorm_kernel_bf16(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        const float* __restrict__ w,
        const size_t rows,
        const size_t cols,
        const float eps = 1e-5f
    ) {
        for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
            const __nv_bfloat16* x_row = x + row * cols;
            __nv_bfloat16* y_row = y + row * cols;

            float sq_sum = 0.f;
            for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
                float val = __bfloat162float(x_row[col]);
                sq_sum += val * val;
            }

            sq_sum = block_reduce_sum(sq_sum);
            float rms = rsqrtf(sq_sum / cols + eps);

            for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
                float val = __bfloat162float(x_row[col]);
                y_row[col] = __float2bfloat16_rn(val * rms * w[col]);
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
}


#endif //NEMOTRON_INFER_REDUCE_CUH

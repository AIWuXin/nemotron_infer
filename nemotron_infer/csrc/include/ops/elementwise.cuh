//
// Created by Administrator on 2026/5/28.
//

#ifndef NEMOTRON_INFER_ELEMENTWISE_CUH
#define NEMOTRON_INFER_ELEMENTWISE_CUH


#include <cuda_bf16.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif


namespace nemotron::ops {
    enum ElementwiseType {
        kElementwiseUnknown = -1,
        kElementwiseAdd = 0,
        kElementwiseMul,
        kElementwiseScale,
        kElementwiseRelu2,
        kElementwiseSilu,
        kElementwiseClampSoftplus,
        // kElementwiseExp
    };


    __device__ __forceinline__ float softplus_fp32(float x) {
        if (x > 20.f) {
            return x;
        }
        if (x < -20.f) {
            return 0.f;
        }

        return __logf(1.f + __expf(x));
    }


    __device__ __forceinline__ float clamp_f32(
        const float x, const float clamp_min, const float clamp_max
    ) {
        return fminf(fmaxf(x, clamp_min), clamp_max);
    }


    /**
     * 该函数使用时最好保证内存16字节对齐
     * @param in0 第一个输入
     * @param in1 第二个输入，可能为空指针
     * @param out0
     * @param scale 缩放操作参数
     * @param bias
     * @param clamp_min
     * @param clamp_max
     **/
    template<ElementwiseType Op>
    __device__ __forceinline__ void elementwise_kernel_fp32(
        const float* __restrict__ in0,
        const float* __restrict__ in1,
        float* __restrict__ out0,
        const size_t size,
        const float scale = 1.f,
        const float* __restrict__ bias = nullptr,
        const float clamp_min = 0.f,
        const float clamp_max = 1.f
    ) {
        const auto in0_vec = reinterpret_cast<const float4*>(in0);
        const auto in1_vec = reinterpret_cast<const float4*>(in1);
        auto out0_vec = reinterpret_cast<float4*>(out0);
        const auto vec_size = size / 4;
        const auto bias_vec = reinterpret_cast<const float4*>(bias);

        for (
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            tid < vec_size; tid += blockDim.x * gridDim.x
        ) {
            const float4 in0_val = in0_vec[tid];
            const float4 in1_val = in1_vec[tid];
            float4 out0_val = {0.f, 0.f, 0.f, 0.f};

            // 向量化处理阶段
            if constexpr (Op == kElementwiseAdd) {
                out0_val.x = in0_val.x + in1_val.x;
                out0_val.y = in0_val.y + in1_val.y;
                out0_val.z = in0_val.z + in1_val.z;
                out0_val.w = in0_val.w + in1_val.w;
            } else if constexpr (Op == kElementwiseMul) {
                out0_val.x = in0_val.x * in1_val.x;
                out0_val.y = in0_val.y * in1_val.y;
                out0_val.z = in0_val.z * in1_val.z;
                out0_val.w = in0_val.w * in1_val.w;
            } else if constexpr (Op == kElementwiseScale) {
                out0_val.x = in0_val.x * scale;
                out0_val.y = in0_val.y * scale;
                out0_val.z = in0_val.z * scale;
                out0_val.w = in0_val.w * scale;
            } else if constexpr (Op == kElementwiseRelu2) {
                out0_val.x = in0_val.x > 0.f ? in0_val.x * in0_val.x : 0.f;
                out0_val.y = in0_val.y > 0.f ? in0_val.y * in0_val.y : 0.f;
                out0_val.z = in0_val.z > 0.f ? in0_val.z * in0_val.z : 0.f;
                out0_val.w = in0_val.w > 0.f ? in0_val.w * in0_val.w : 0.f;
            } else if constexpr (Op == kElementwiseSilu) {
                out0_val.x = in0_val.x / (1.f + __expf(-in0_val.x));
                out0_val.y = in0_val.y / (1.f + __expf(-in0_val.y));
                out0_val.z = in0_val.z / (1.f + __expf(-in0_val.z));
                out0_val.w = in0_val.w / (1.f + __expf(-in0_val.w));
            } else if constexpr (Op == kElementwiseClampSoftplus) {
                const auto bias_val = bias_vec[tid];
                out0_val.x = clamp_f32(
                    softplus_fp32(in0_val.x + bias_val.x),
                    clamp_min, clamp_max
                );
                out0_val.y = clamp_f32(
                    softplus_fp32(in0_val.y + bias_val.y),
                    clamp_min, clamp_max
                );
                out0_val.z = clamp_f32(
                    softplus_fp32(in0_val.z + bias_val.z),
                    clamp_min, clamp_max
                );
                out0_val.w = clamp_f32(
                    softplus_fp32(in0_val.w + bias_val.w),
                    clamp_min, clamp_max
                );
            }
            out0_vec[tid] = out0_val;
        }

        // 处理尾数阶段
        if (blockIdx.x == 0) {
            for (
                size_t tid = vec_size * 4 + threadIdx.x;
                tid < size; tid += blockDim.x
            ) {
                if constexpr (Op == kElementwiseAdd) {
                    out0[tid] = in0[tid] + in1[tid];
                } else if constexpr (Op == kElementwiseMul) {
                    out0[tid] = in0[tid] * in1[tid];
                } else if constexpr (Op == kElementwiseScale) {
                    out0[tid] = in0[tid] * scale;
                } else if constexpr (Op == kElementwiseRelu2) {
                    out0[tid] = in0[tid] > 0.f ? in0[tid] * in0[tid] : 0.f;
                } else if constexpr (Op == kElementwiseSilu) {
                    out0[tid] = in0[tid] / (1.f + __expf(-in0[tid]));
                } else if constexpr (Op == kElementwiseClampSoftplus) {
                    out0[tid] = clamp_f32(
                        softplus_fp32(in0[tid] + bias[tid]),
                        clamp_min, clamp_max
                    );
                }
            }
        }
    }


    template<ElementwiseType Op>
    __global__ void elementwise_launch_fp32(
        const float* __restrict__ in0,
        const float* __restrict__ in1,
        float* __restrict__ out0,
        const size_t size,
        const float scale = 1.f,
        const float* bias = nullptr,
        const float clamp_min = 0.f,
        const float clamp_max = 1.f
    ) {
        elementwise_kernel_fp32<Op>(
            in0, in1, out0, size, scale,
            bias, clamp_min, clamp_max
        );
    }


    template<ElementwiseType ops_type>
    __host__ void elementwise_ops_fp32(
        float* in0,
        float* in1,
        float* out0,
        const size_t size,
        const float scale = 1.f,
        const float* bias = nullptr,
        const float clamp_min = 0.f,
        const float clamp_max = 1.f,
        cudaStream_t stream = nullptr
    ) {
        constexpr size_t block_dimx = 1024;
        const size_t grid_dimx = (size + block_dimx - 1) / block_dimx;

        elementwise_launch_fp32<ops_type><<<grid_dimx, block_dimx, 0, stream>>>(
            in0, in1, out0, size, scale, bias, clamp_min, clamp_max
        );
    }


    /**
     * @attention bf16类型的算子必须满足十六字节对齐
     * 不会处理尾数，使用前请注意
     **/
    template<ElementwiseType Op>
    __device__ __forceinline__ void elementwise_kernel_bf16(
        const __nv_bfloat16* __restrict__ in0,
        const __nv_bfloat16* __restrict__ in1,
        __nv_bfloat16* __restrict__ out0,
        const size_t size,
        const float scale = 1.f,
        const __nv_bfloat16* __restrict__ bias = nullptr,
        const float clamp_min = 0.f,
        const float clamp_max = 1.f
    ) {
        const auto in0_vec = reinterpret_cast<const float4*>(in0);
        const auto in1_vec = reinterpret_cast<const float4*>(in1);
        auto out0_vec = reinterpret_cast<float4*>(out0);
        // 先使用float4读取，然后解包成4*2个bfloat16数据
        const auto vec_size = size / 8;
        const auto bias_vec = reinterpret_cast<const float4*>(bias);
        const auto bf_scale = __float2bfloat162_rn(scale);
        __nv_bfloat162 bf_one  = __float2bfloat162_rn(1.0f);
        __nv_bfloat162 bf_zero = __float2bfloat162_rn(0.0f);
        __nv_bfloat162 out_x{}, out_y{}, out_z{}, out_w{};

        for (
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            tid < vec_size; tid += blockDim.x * gridDim.x
        ) {
            const float4 in0_val = in0_vec[tid];
            const float4 in1_val = in1_vec[tid];

            const auto& in0_bf_val_x = reinterpret_cast<const __nv_bfloat162&>(in0_val.x);
            const auto& in0_bf_val_y = reinterpret_cast<const __nv_bfloat162&>(in0_val.y);
            const auto& in0_bf_val_z = reinterpret_cast<const __nv_bfloat162&>(in0_val.z);
            const auto& in0_bf_val_w = reinterpret_cast<const __nv_bfloat162&>(in0_val.w);

            const auto& in1_bf_val_x = reinterpret_cast<const __nv_bfloat162&>(in1_val.x);
            const auto& in1_bf_val_y = reinterpret_cast<const __nv_bfloat162&>(in1_val.y);
            const auto& in1_bf_val_z = reinterpret_cast<const __nv_bfloat162&>(in1_val.z);
            const auto& in1_bf_val_w = reinterpret_cast<const __nv_bfloat162&>(in1_val.w);

            if constexpr (Op == kElementwiseAdd) {
                out_x = __hadd2(in0_bf_val_x, in1_bf_val_x);
                out_y = __hadd2(in0_bf_val_y, in1_bf_val_y);
                out_z = __hadd2(in0_bf_val_z, in1_bf_val_z);
                out_w = __hadd2(in0_bf_val_w, in1_bf_val_w);
            } else if constexpr (Op == kElementwiseMul) {
                out_x = __hmul2(in0_bf_val_x, in1_bf_val_x);
                out_y = __hmul2(in0_bf_val_y, in1_bf_val_y);
                out_z = __hmul2(in0_bf_val_z, in1_bf_val_z);
                out_w = __hmul2(in0_bf_val_w, in1_bf_val_w);
            } else if constexpr (Op == kElementwiseScale) {
                out_x = __hmul2(in0_bf_val_x, bf_scale);
                out_y = __hmul2(in0_bf_val_y, bf_scale);
                out_z = __hmul2(in0_bf_val_z, bf_scale);
                out_w = __hmul2(in0_bf_val_w, bf_scale);
            } else if constexpr (Op == kElementwiseRelu2) {
                out_x = __hfma2_relu(
                    in0_bf_val_x, bf_one, bf_zero
                );
                out_y = __hfma2_relu(
                    in0_bf_val_y, bf_one, bf_zero
                );
                out_z = __hfma2_relu(
                    in0_bf_val_z, bf_one, bf_zero
                );
                out_w = __hfma2_relu(
                    in0_bf_val_w, bf_one, bf_zero
                );
                out_x = __hmul2(out_x, out_x);
                out_y = __hmul2(out_y, out_y);
                out_z = __hmul2(out_z, out_z);
                out_w = __hmul2(out_w, out_w);
            } else if constexpr (Op == kElementwiseSilu) {
#ifdef USE_CUDNN

#else
                auto in0_f_val_x = __bfloat1622float2(in0_bf_val_x);
                auto in0_f_val_y = __bfloat1622float2(in0_bf_val_y);
                auto in0_f_val_z = __bfloat1622float2(in0_bf_val_z);
                auto in0_f_val_w = __bfloat1622float2(in0_bf_val_w);

                in0_f_val_x.x = in0_f_val_x.x / (1.f + __expf(-in0_f_val_x.x));
                in0_f_val_x.y = in0_f_val_x.y / (1.f + __expf(-in0_f_val_x.y));
                in0_f_val_y.x = in0_f_val_y.x / (1.f + __expf(-in0_f_val_y.x));
                in0_f_val_y.y = in0_f_val_y.y / (1.f + __expf(-in0_f_val_y.y));
                in0_f_val_z.x = in0_f_val_z.x / (1.f + __expf(-in0_f_val_z.x));
                in0_f_val_z.y = in0_f_val_z.y / (1.f + __expf(-in0_f_val_z.y));
                in0_f_val_w.x = in0_f_val_w.x / (1.f + __expf(-in0_f_val_w.x));
                in0_f_val_w.y = in0_f_val_w.y / (1.f + __expf(-in0_f_val_w.y));

                out_x = __float22bfloat162_rn(in0_f_val_x);
                out_y = __float22bfloat162_rn(in0_f_val_y);
                out_z = __float22bfloat162_rn(in0_f_val_z);
                out_w = __float22bfloat162_rn(in0_f_val_w);
#endif
            } else if constexpr (Op == kElementwiseClampSoftplus) {
                const float4 bias_val = bias_vec[tid];

                auto in0_f_val_x = __bfloat1622float2(in0_bf_val_x);
                auto in0_f_val_y = __bfloat1622float2(in0_bf_val_y);
                auto in0_f_val_z = __bfloat1622float2(in0_bf_val_z);
                auto in0_f_val_w = __bfloat1622float2(in0_bf_val_w);

                const auto& bias_bf_val_x = reinterpret_cast<const __nv_bfloat162&>(bias_val.x);
                const auto& bias_bf_val_y = reinterpret_cast<const __nv_bfloat162&>(bias_val.y);
                const auto& bias_bf_val_z = reinterpret_cast<const __nv_bfloat162&>(bias_val.z);
                const auto& bias_bf_val_w = reinterpret_cast<const __nv_bfloat162&>(bias_val.w);

                auto bias_f_val_x = __bfloat1622float2(bias_bf_val_x);
                auto bias_f_val_y = __bfloat1622float2(bias_bf_val_y);
                auto bias_f_val_z = __bfloat1622float2(bias_bf_val_z);
                auto bias_f_val_w = __bfloat1622float2(bias_bf_val_w);

                in0_f_val_x.x = clamp_f32(
                    softplus_fp32(in0_f_val_x.x + bias_f_val_x.x),
                    clamp_min, clamp_max
                );
                in0_f_val_x.y = clamp_f32(
                    softplus_fp32(in0_f_val_x.y + bias_f_val_x.y),
                    clamp_min, clamp_max
                );
                in0_f_val_y.x = clamp_f32(
                    softplus_fp32(in0_f_val_y.x + bias_f_val_y.x),
                    clamp_min, clamp_max
                );
                in0_f_val_y.y = clamp_f32(
                    softplus_fp32(in0_f_val_y.y + bias_f_val_y.y),
                    clamp_min, clamp_max
                );
                in0_f_val_z.x = clamp_f32(
                    softplus_fp32(in0_f_val_z.x + bias_f_val_z.x),
                    clamp_min, clamp_max
                );
                in0_f_val_z.y = clamp_f32(
                    softplus_fp32(in0_f_val_z.y + bias_f_val_z.y),
                    clamp_min, clamp_max
                );
                in0_f_val_w.x = clamp_f32(
                    softplus_fp32(in0_f_val_w.x + bias_f_val_w.x),
                    clamp_min, clamp_max
                );
                in0_f_val_w.y = clamp_f32(
                    softplus_fp32(in0_f_val_w.y + bias_f_val_w.y),
                    clamp_min, clamp_max
                );

                out_x = __float22bfloat162_rn(in0_f_val_x);
                out_y = __float22bfloat162_rn(in0_f_val_y);
                out_z = __float22bfloat162_rn(in0_f_val_z);
                out_w = __float22bfloat162_rn(in0_f_val_w);
            }
            out0_vec[tid] = make_float4(
                reinterpret_cast<const float&>(out_x),
                reinterpret_cast<const float&>(out_y),
                reinterpret_cast<const float&>(out_z),
                reinterpret_cast<const float&>(out_w)
            );
        }
    }

    template<ElementwiseType Op>
    __global__ void elementwise_launch_bf16(
        const __nv_bfloat16* __restrict__ in0,
        const __nv_bfloat16* __restrict__ in1,
        __nv_bfloat16* __restrict__ out0,
        const size_t size,
        const float scale = 1.f,
        const __nv_bfloat16* __restrict__ bias = nullptr,
        const float clamp_min = 0.f,
        const float clamp_max = 1.f
    ) {
        elementwise_kernel_bf16<Op>(
            in0, in1, out0, size, scale,
            bias, clamp_min, clamp_max
        );
    }

    template<ElementwiseType ops_type>
    __host__ void elementwise_ops_bf16(
        const __nv_bfloat16* __restrict__ in0,
        const __nv_bfloat16* __restrict__ in1,
        __nv_bfloat16* __restrict__ out0,
        const size_t size,
        const float scale = 1.f,
        const __nv_bfloat16* __restrict__ bias = nullptr,
        const float clamp_min = 0.f,
        const float clamp_max = 1.f,
        cudaStream_t stream = nullptr
    ) {
        constexpr size_t block_dimx = 1024;
        const size_t grid_dimx = (size + block_dimx - 1) / block_dimx;

        elementwise_launch_bf16<ops_type><<<grid_dimx, block_dimx, 0, stream>>>(
            in0, in1, out0, size, scale, bias, clamp_min, clamp_max
        );
    }
}


#endif //NEMOTRON_INFER_ELEMENTWISE_CUH

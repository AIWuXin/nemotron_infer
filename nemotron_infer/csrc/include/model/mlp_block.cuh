//
// mlp_block.cuh — Nemotron-H MLP 类型 DecoderBlock（prefill + decode）
// ===========================================================================
// 组装单层 `NemotronHBlock`(block_type=mlp)：
//   out = input + mixer( rmsnorm(input) )            ← pre-norm 残差
// mixer = NemotronHMLP.forward:
//   up_proj → relu²(ReLU(x)²) → down_proj            ← 无门控
//
// 形状（真实模型）：HIDDEN=3136, INTER=12544。
// relu2 复用 elementwise kElementwiseRelu2（out = x>0 ? x² : 0）。
// 精度路径与 mamba_block 对齐：
//   fp32 参考；bf16(gemm 内部 TF32)；fp8(in/out matmul 走原生 e4m3)。
// decode 与 prefill 数学完全一致（MLP 逐 token 无状态），仅 M=B。
// ===========================================================================
#ifndef NEMOTRON_INFER_MODEL_MLP_BLOCK_CUH
#define NEMOTRON_INFER_MODEL_MLP_BLOCK_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tensor/tensor.h"
#include "ops/reduce.cuh"
#include "ops/gemm.cuh"
#include "ops/elementwise.cuh"

namespace nemotron::model {

// ===========================================================================
// FP32 参考路径
// ===========================================================================
struct MLPBlockWeightsFP32 {
    const float* block_norm_w;   // [HIDDEN]          外层 block 的 norm.weight
    const float* up_proj_w;      // [INTER, HIDDEN]   mixer.up_proj.weight
    const float* down_proj_w;    // [HIDDEN, INTER]   mixer.down_proj.weight
};

template<int HIDDEN, int INTER>
inline void mlp_block_forward_fp32(
    const float* input,          // [M, HIDDEN]
    const MLPBlockWeightsFP32& w,
    float* out,                  // [M, HIDDEN]
    int M,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    constexpr float EPS = 1e-5f;

    auto normed = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));
    auto up     = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto act    = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto down   = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));

    // 1. pre-norm RMSNorm
    rmsnorm_fp32(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);
    // 2. up_proj
    gemm_fp32(normed.data_, w.up_proj_w, up.data_, M, INTER, HIDDEN, true, stream);
    // 3. relu²（in1 未用，传 up 占位）
    elementwise_ops_fp32<kElementwiseRelu2>(
        up.data_, up.data_, act.data_, (size_t)M * INTER,
        1.f, nullptr, 0.f, 1.f, stream);
    // 4. down_proj
    gemm_fp32(act.data_, w.down_proj_w, down.data_, M, HIDDEN, INTER, true, stream);
    // 5. 残差
    elementwise_ops_fp32<kElementwiseAdd>(
        const_cast<float*>(input), down.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// BF16 路径（gemm 内部 TF32；norm/elementwise bf16 IO fp32 累加）
//   权重：up/down 为 bf16（matmul）；norm.weight fp32。
// ===========================================================================
struct MLPBlockWeightsBF16 {
    const float*         block_norm_w;   // [HIDDEN]          fp32
    const __nv_bfloat16* up_proj_w;      // [INTER, HIDDEN]   bf16
    const __nv_bfloat16* down_proj_w;    // [HIDDEN, INTER]   bf16
};

template<int HIDDEN, int INTER>
inline void mlp_block_forward_bf16(
    const __nv_bfloat16* input,          // [M, HIDDEN]
    const MLPBlockWeightsBF16& w,
    __nv_bfloat16* out,                  // [M, HIDDEN]
    int M,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using bf16 = __nv_bfloat16;
    constexpr float EPS = 1e-5f;

    auto normed = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto up     = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto act    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto down   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));

    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);
    gemm_bf16(normed.data_, w.up_proj_w, up.data_, M, INTER, HIDDEN, stream);
    elementwise_ops_bf16<kElementwiseRelu2>(
        up.data_, up.data_, act.data_, (size_t)M * INTER,
        1.f, nullptr, 0.f, 1.f, stream);
    gemm_bf16(act.data_, w.down_proj_w, down.data_, M, HIDDEN, INTER, stream);
    elementwise_ops_bf16<kElementwiseAdd>(
        input, down.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// FP8 路径（up/down matmul 走原生 e4m3；激活设备端动态量化 per-tensor）
//   relu²/norm/残差 与 bf16 一致。
// ===========================================================================
struct MLPBlockWeightsFP8 {
    const float*         block_norm_w;     // [HIDDEN]          fp32
    const __nv_fp8_e4m3* up_proj_w;        // [INTER, HIDDEN]   e4m3
    const float*         up_proj_wscale;   // [INTER]           per-row
    const __nv_fp8_e4m3* down_proj_w;      // [HIDDEN, INTER]   e4m3
    const float*         down_proj_wscale; // [HIDDEN]          per-row
};

template<int HIDDEN, int INTER>
inline void mlp_block_forward_fp8(
    const __nv_bfloat16* input,          // [M, HIDDEN]
    const MLPBlockWeightsFP8& w,
    __nv_bfloat16* out,                  // [M, HIDDEN]
    int M,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using bf16 = __nv_bfloat16;
    using fp8  = __nv_fp8_e4m3;
    constexpr float EPS = 1e-5f;
    constexpr size_t WS_BYTES = 32ull * 1024 * 1024;

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto norm_fp8 = allocate_tensor<fp8>(TensorShape::make_2d(M, HIDDEN));
    auto up       = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto act      = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto act_fp8  = allocate_tensor<fp8>(TensorShape::make_2d(M, INTER));
    auto down     = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto xscale1  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto xscale2  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto amax     = allocate_tensor<unsigned int>(TensorShape::make_1d(1));
    auto ws       = allocate_tensor<char>(TensorShape::make_1d((int64_t)WS_BYTES));

    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    // up_proj: 量化 normed → fp8
    quantize_activation_fp8(normed.data_, norm_fp8.data_, xscale1.data_,
                            amax.data_, (size_t)M * HIDDEN, stream);
    gemm_fp8(norm_fp8.data_, w.up_proj_w, up.data_, xscale1.data_, w.up_proj_wscale,
             M, INTER, HIDDEN, ws.data_, WS_BYTES, stream);

    elementwise_ops_bf16<kElementwiseRelu2>(
        up.data_, up.data_, act.data_, (size_t)M * INTER,
        1.f, nullptr, 0.f, 1.f, stream);

    // down_proj: 量化 act → fp8
    quantize_activation_fp8(act.data_, act_fp8.data_, xscale2.data_,
                            amax.data_, (size_t)M * INTER, stream);
    gemm_fp8(act_fp8.data_, w.down_proj_w, down.data_, xscale2.data_, w.down_proj_wscale,
             M, HIDDEN, INTER, ws.data_, WS_BYTES, stream);

    elementwise_ops_bf16<kElementwiseAdd>(
        input, down.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

}  // namespace nemotron::model

#endif  // NEMOTRON_INFER_MODEL_MLP_BLOCK_CUH

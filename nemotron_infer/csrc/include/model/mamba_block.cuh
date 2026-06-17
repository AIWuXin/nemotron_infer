//
// mamba_block.cuh — Nemotron-H Mamba 类型 DecoderBlock（prefill, fp32）
// ===========================================================================
// 组装单层 `NemotronHBlock`(block_type=mamba)：
//   out = input + mixer( rmsnorm(input) )          ← pre-norm 残差
// mixer = NemotronHMamba2Mixer.torch_forward(prefill, no cache):
//   in_proj → split[gate|xBC|dt] → conv1d+SiLU → split[x|B|C]
//          → SSD scan(+D skip) → gated RMSNorm(gate) → out_proj
//
// 设计要点：
//  - in_proj split 用「权重按输出行切片 → 3 个连续 GEMM」，避免 strided 激活视图
//    （in_proj_w 行主序 [PROJ,HIDDEN]，输出行区间即连续子块）。
//  - conv→x/B/C split 用 cudaMemcpy2D 从 [M,CONV_DIM] strided 切出连续 x/B/C。
//  - B/C 不物化 repeat：SSD kernel 内部按 group=h/(H/G) 索引。
//  - dt 离散化在 SSD kernel 内 softplus+clamp，clamp=(0,FLT_MAX)=模型 dt_limit。
//  - 全程 fp32（精度表：SSD scan 必须 fp32）。这是正确优先的第一版，未做融合/复用缓冲。
// ===========================================================================
#ifndef NEMOTRON_INFER_MODEL_MAMBA_BLOCK_CUH
#define NEMOTRON_INFER_MODEL_MAMBA_BLOCK_CUH

#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tensor/tensor.h"
#include "ops/reduce.cuh"
#include "ops/gemm.cuh"
#include "ops/elementwise.cuh"
#include "ops/mamba2/causal_conv1d.cuh"
#include "ops/mamba2/ssd_scan.cuh"
#include "ops/mamba2/ssd_scan_chunked.cuh"
#include "ops/mamba2/ssm.cuh"

namespace nemotron::model {

// 全 device 指针（行主序，命名同 HF backbone.layers.{i}.*）
struct Mamba2BlockWeightsFP32 {
    const float* block_norm_w;   // [HIDDEN]              外层 block 的 norm.weight
    const float* in_proj_w;      // [PROJ, HIDDEN]        mixer.in_proj.weight
    const float* conv1d_w;       // [CONV_DIM, CONV_K]    mixer.conv1d.weight (展平自 [conv_dim,1,K])
    const float* conv1d_b;       // [CONV_DIM]            mixer.conv1d.bias
    const float* A_log;          // [H]
    const float* D;              // [H]
    const float* dt_bias;        // [H]
    const float* gnorm_w;        // [INTER]               mixer.norm.weight (gated)
    const float* out_proj_w;     // [HIDDEN, INTER]       mixer.out_proj.weight
};

// 维度走编译期模板（与底层算子模板对齐）；运行期只传 B,S。
template<int HIDDEN, int H, int P, int N, int G, int CONV_K = 4>
inline void mamba_block_forward_fp32(
    const float* input,          // [B*S, HIDDEN]
    const Mamba2BlockWeightsFP32& w,
    float* out,                  // [B*S, HIDDEN]
    int B, int S,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::mamba2;

    constexpr int INTER      = H * P;                    // intermediate
    constexpr int CONV_DIM   = INTER + 2 * G * N;        // conv 通道
    constexpr int PROJ       = INTER + CONV_DIM + H;     // in_proj 输出
    constexpr int GROUP_SIZE = INTER / G;                // gated norm 每组宽度
    const int M = B * S;
    constexpr float EPS = 1e-5f;

    // 中间缓冲（BumpAllocator；测试每次 SetUp reset，单次前向无需逐块释放）
    auto normed   = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));
    auto gate     = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto xbc      = allocate_tensor<float>(TensorShape::make_2d(M, CONV_DIM));
    auto dt       = allocate_tensor<float>(TensorShape::make_2d(M, H));
    auto xbc_conv = allocate_tensor<float>(TensorShape::make_2d(M, CONV_DIM));
    auto x_buf    = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto B_buf    = allocate_tensor<float>(TensorShape::make_2d(M, G * N));
    auto C_buf    = allocate_tensor<float>(TensorShape::make_2d(M, G * N));
    auto y_buf    = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto gnormed  = allocate_tensor<float>(TensorShape::make_2d(M, INTER));
    auto mixer    = allocate_tensor<float>(TensorShape::make_2d(M, HIDDEN));

    // 1. pre-norm RMSNorm
    rmsnorm_fp32(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    // 2. in_proj split = 3 个连续 GEMM（W 按输出行切片，仍连续）
    gemm_fp32(normed.data_, w.in_proj_w,
              gate.data_, M, INTER, HIDDEN, true, stream);
    gemm_fp32(normed.data_, w.in_proj_w + (size_t)INTER * HIDDEN,
              xbc.data_,  M, CONV_DIM, HIDDEN, true, stream);
    gemm_fp32(normed.data_, w.in_proj_w + (size_t)(INTER + CONV_DIM) * HIDDEN,
              dt.data_,   M, H, HIDDEN, true, stream);

    // 3. depthwise causal conv1d + SiLU（内部含 SiLU）
    causal_conv1d_prefill_fp32<64>(xbc.data_, w.conv1d_w, w.conv1d_b,
                                   xbc_conv.data_, nullptr, CONV_DIM, S, B, stream);

    // 4. split xBC → x | B | C（[M,CONV_DIM] strided → 连续，cudaMemcpy2D）
    cudaMemcpy2DAsync(x_buf.data_, (size_t)INTER * sizeof(float),
                      xbc_conv.data_, (size_t)CONV_DIM * sizeof(float),
                      (size_t)INTER * sizeof(float), M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(B_buf.data_, (size_t)(G * N) * sizeof(float),
                      xbc_conv.data_ + INTER, (size_t)CONV_DIM * sizeof(float),
                      (size_t)(G * N) * sizeof(float), M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(C_buf.data_, (size_t)(G * N) * sizeof(float),
                      xbc_conv.data_ + INTER + G * N, (size_t)CONV_DIM * sizeof(float),
                      (size_t)(G * N) * sizeof(float), M, cudaMemcpyDeviceToDevice, stream);

    // 5. SSD scan（B/C 内部 group 索引；dt_limit=(0,inf)）
    ssd_scan_prefill_fp32<H, P, N>(
        x_buf.data_, dt.data_, w.A_log, B_buf.data_, C_buf.data_,
        w.D, w.dt_bias, y_buf.data_, /*ssm_state_out*/ nullptr,
        B, S, G, /*dt_min*/ 0.f, /*dt_max*/ FLT_MAX, stream);

    // 6. gated RMSNorm: RMSNorm(y·SiLU(gate))·w（grouped）
    rmsnorm_gated_fp32<G>(y_buf.data_, gnormed.data_, w.gnorm_w, gate.data_,
                          M, INTER, GROUP_SIZE, EPS, stream);

    // 7. out_proj
    gemm_fp32(gnormed.data_, w.out_proj_w, mixer.data_, M, HIDDEN, INTER, true, stream);

    // 8. 残差：out = input + mixer
    elementwise_ops_fp32<kElementwiseAdd>(
        const_cast<float*>(input), mixer.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// BF16 混合精度路径（精度表 v1）
//   激活全程 bf16；in_proj/out_proj 走 gemm_bf16(内部 TF32 Tensor Core)；
//   conv1d/gated-norm/elementwise 走 bf16 算子(fp32 累加)；
//   SSD scan 走 bf16-IO 模板内核(内部 fp32 计算，红线不破)。
//   权重精度：仅 in_proj_w/out_proj_w 为 bf16（matmul 需要）；
//   conv w/b、A_log/D/dt_bias、两处 norm.weight 仍 fp32（对应算子均收 fp32 权重）。
// ===========================================================================
struct Mamba2BlockWeightsBF16 {
    const float*         block_norm_w;   // [HIDDEN]            fp32
    const __nv_bfloat16* in_proj_w;      // [PROJ, HIDDEN]     bf16
    const float*         conv1d_w;       // [CONV_DIM, CONV_K] fp32
    const float*         conv1d_b;       // [CONV_DIM]         fp32
    const float*         A_log;          // [H]                fp32
    const float*         D;              // [H]                fp32
    const float*         dt_bias;        // [H]                fp32
    const float*         gnorm_w;        // [INTER]            fp32
    const __nv_bfloat16* out_proj_w;     // [HIDDEN, INTER]    bf16
};

template<int HIDDEN, int H, int P, int N, int G, int CONV_K = 4>
inline void mamba_block_forward_bf16(
    const __nv_bfloat16* input,          // [B*S, HIDDEN]
    const Mamba2BlockWeightsBF16& w,
    __nv_bfloat16* out,                  // [B*S, HIDDEN]
    int B, int S,
    float* conv_state_out = nullptr,     // [B, CONV_DIM, CONV_K-1] 末态（供 decode 续接）
    float* ssm_state_out  = nullptr,     // [B, H, D, N] 末态（供 decode 续接）
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::mamba2;
    using bf16 = __nv_bfloat16;

    constexpr int INTER      = H * P;
    constexpr int CONV_DIM   = INTER + 2 * G * N;
    constexpr int PROJ       = INTER + CONV_DIM + H;
    constexpr int GROUP_SIZE = INTER / G;
    const int M = B * S;
    constexpr float EPS = 1e-5f;
    constexpr size_t BF = sizeof(bf16);

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto gate     = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto xbc      = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto dt       = allocate_tensor<bf16>(TensorShape::make_2d(M, H));
    auto xbc_conv = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto x_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto B_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto C_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto y_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto gnormed  = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto mixer    = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));

    // 1. pre-norm RMSNorm (bf16 IO, fp32 累加)
    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    // 2. in_proj split = 3 个连续 gemm_bf16（W 按输出行切片，仍连续；内部 TF32）
    gemm_bf16(normed.data_, w.in_proj_w,
              gate.data_, M, INTER, HIDDEN, stream);
    gemm_bf16(normed.data_, w.in_proj_w + (size_t)INTER * HIDDEN,
              xbc.data_,  M, CONV_DIM, HIDDEN, stream);
    gemm_bf16(normed.data_, w.in_proj_w + (size_t)(INTER + CONV_DIM) * HIDDEN,
              dt.data_,   M, H, HIDDEN, stream);

    // 3. depthwise causal conv1d + SiLU (bf16 IO, fp32 累加；conv w/b fp32)
    //    conv_state_out: 末 K-1 个原始输入（pre-conv），布局已与 conv1d_decode 一致
    causal_conv1d_prefill_bf16<64>(xbc.data_, w.conv1d_w, w.conv1d_b,
                                   xbc_conv.data_, conv_state_out, CONV_DIM, S, B, stream);

    // 4. split xBC → x | B | C（bf16，pitch 按 bf16 大小）
    cudaMemcpy2DAsync(x_buf.data_, (size_t)INTER * BF,
                      xbc_conv.data_, (size_t)CONV_DIM * BF,
                      (size_t)INTER * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(B_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(C_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER + G * N, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);

    // 5. SSD scan（chunked TF32：分块矩阵乘，比串行快 ~1.2x；cumsum/exp fp32，红线不破）
    //    ssm_state_out: [B,H,D,N] 末态，与 ssm_decode 的 state 布局一致
    ssd_scan_chunked_prefill_bf16<H, P, N, 128>(
        x_buf.data_, dt.data_, w.A_log, B_buf.data_, C_buf.data_,
        w.D, w.dt_bias, y_buf.data_, ssm_state_out,
        B, S, G, /*dt_min*/ 0.f, /*dt_max*/ FLT_MAX, stream);

    // 6. gated RMSNorm: RMSNorm(y·SiLU(gate))·w（grouped；bf16 IO, fp32 累加）
    rmsnorm_gated_bf16<G>(y_buf.data_, gnormed.data_, w.gnorm_w, gate.data_,
                          M, INTER, GROUP_SIZE, EPS, stream);

    // 7. out_proj (gemm_bf16)
    gemm_bf16(gnormed.data_, w.out_proj_w, mixer.data_, M, HIDDEN, INTER, stream);

    // 8. 残差：out = input + mixer (bf16)
    elementwise_ops_bf16<kElementwiseAdd>(
        input, mixer.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// BF16 decode：单步生成（每序列 1 token），续接 prefill 末态
//   数据流同 prefill，但 conv/scan 走单步 decode 算子并就地更新状态：
//     rmsnorm → in_proj(gemv×3) → conv1d_decode(更新 conv_state)
//       → split x|B|C → ssm_decode(更新 ssm_state) → gated_norm → out_proj → 残差
//   conv_state[B,CONV_DIM,K-1] / ssm_state[B,H,D,N] 由 prefill 产出，此处 in/out。
//   Mamba 的关键性质：每 token O(1) 状态更新，与上下文长度无关。
// ===========================================================================
template<int HIDDEN, int H, int P, int N, int G, int CONV_K = 4>
inline void mamba_block_decode_bf16(
    const __nv_bfloat16* input,          // [B, HIDDEN] 单 token
    const Mamba2BlockWeightsBF16& w,
    float* conv_state,                   // [B, CONV_DIM, CONV_K-1] in/out
    float* ssm_state,                    // [B, H, D, N] in/out
    __nv_bfloat16* out,                  // [B, HIDDEN]
    int B,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::mamba2;
    using bf16 = __nv_bfloat16;

    constexpr int INTER      = H * P;
    constexpr int CONV_DIM   = INTER + 2 * G * N;
    constexpr int GROUP_SIZE = INTER / G;
    const int M = B;                     // 每序列 1 token
    constexpr float EPS = 1e-5f;
    constexpr size_t BF = sizeof(bf16);

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto gate     = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto xbc      = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto dt       = allocate_tensor<bf16>(TensorShape::make_2d(M, H));
    auto xbc_conv = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto x_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto B_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto C_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto y_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto gnormed  = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto mixer    = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));

    // 1. pre-norm RMSNorm
    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    // 2. in_proj split = 3 个 gemv（M=1，gemm_bf16 退化为矩阵-向量）
    gemm_bf16(normed.data_, w.in_proj_w, gate.data_, M, INTER, HIDDEN, stream);
    gemm_bf16(normed.data_, w.in_proj_w + (size_t)INTER * HIDDEN,
              xbc.data_, M, CONV_DIM, HIDDEN, stream);
    gemm_bf16(normed.data_, w.in_proj_w + (size_t)(INTER + CONV_DIM) * HIDDEN,
              dt.data_, M, H, HIDDEN, stream);

    // 3. conv1d decode（更新 conv_state，内部 SiLU）
    causal_conv1d_decode_bf16<CONV_K>(xbc.data_, w.conv1d_w, w.conv1d_b,
                                      xbc_conv.data_, conv_state, CONV_DIM, B, stream);

    // 4. split xBC → x | B | C（[B,CONV_DIM] strided → 连续；B=1 即单行拷贝）
    cudaMemcpy2DAsync(x_buf.data_, (size_t)INTER * BF,
                      xbc_conv.data_, (size_t)CONV_DIM * BF,
                      (size_t)INTER * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(B_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(C_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER + G * N, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);

    // 5. ssm decode 单步（更新 ssm_state；grid=B*H, block=128）
    ssm_decode_bf16<H, P, N><<<dim3(B * H), 128, 0, stream>>>(
        x_buf.data_, dt.data_, w.A_log, B_buf.data_, C_buf.data_,
        w.D, ssm_state, y_buf.data_, w.dt_bias, B, G, 0.f, FLT_MAX);

    // 6. gated RMSNorm
    rmsnorm_gated_bf16<G>(y_buf.data_, gnormed.data_, w.gnorm_w, gate.data_,
                          M, INTER, GROUP_SIZE, EPS, stream);

    // 7. out_proj
    gemm_bf16(gnormed.data_, w.out_proj_w, mixer.data_, M, HIDDEN, INTER, stream);

    // 8. 残差
    elementwise_ops_bf16<kElementwiseAdd>(
        input, mixer.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// FP8 GEMM 混合精度路径（精度表 v2）
//   与 bf16 路径完全相同，唯独 in_proj/out_proj 走原生 fp8(e4m3) gemm：
//     设备端动态量化激活(per-tensor scale) → gemm_fp8(累加 fp32 + epilogue 列 scale)。
//   权重预量化到 e4m3 + per-row(输出通道) scale（加载期一次性，离线/host 完成）。
//   in_proj 单矩阵：量化 normed 一次，x_fp8/x_scale 复用给 gate/xBC/dt 三个行切片。
//   conv1d/gated-norm/elementwise/SSD scan 与 bf16 路径一致（scan 内部 fp32）。
// ===========================================================================
struct Mamba2BlockWeightsFP8 {
    const float*         block_norm_w;     // [HIDDEN]            fp32
    const __nv_fp8_e4m3* in_proj_w;        // [PROJ, HIDDEN]     e4m3
    const float*         in_proj_wscale;   // [PROJ]             per-row
    const float*         conv1d_w;         // [CONV_DIM, CONV_K] fp32
    const float*         conv1d_b;         // [CONV_DIM]         fp32
    const float*         A_log;            // [H]                fp32
    const float*         D;                // [H]                fp32
    const float*         dt_bias;          // [H]                fp32
    const float*         gnorm_w;          // [INTER]            fp32
    const __nv_fp8_e4m3* out_proj_w;       // [HIDDEN, INTER]    e4m3
    const float*         out_proj_wscale;  // [HIDDEN]           per-row
};

template<int HIDDEN, int H, int P, int N, int G, int CONV_K = 4>
inline void mamba_block_forward_fp8(
    const __nv_bfloat16* input,          // [B*S, HIDDEN]
    const Mamba2BlockWeightsFP8& w,
    __nv_bfloat16* out,                  // [B*S, HIDDEN]
    int B, int S,
    float* conv_state_out = nullptr,     // [B, CONV_DIM, CONV_K-1] 末态（供 decode 续接）
    float* ssm_state_out  = nullptr,     // [B, H, D, N] 末态（供 decode 续接）
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::mamba2;
    using bf16 = __nv_bfloat16;
    using fp8  = __nv_fp8_e4m3;

    constexpr int INTER      = H * P;
    constexpr int CONV_DIM   = INTER + 2 * G * N;
    constexpr int PROJ       = INTER + CONV_DIM + H;
    constexpr int GROUP_SIZE = INTER / G;
    const int M = B * S;
    constexpr float EPS = 1e-5f;
    constexpr size_t BF = sizeof(bf16);
    constexpr size_t WS_BYTES = 32ull * 1024 * 1024;   // cuBLASLt workspace

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto norm_fp8 = allocate_tensor<fp8>(TensorShape::make_2d(M, HIDDEN));
    auto gate     = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto xbc      = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto dt       = allocate_tensor<bf16>(TensorShape::make_2d(M, H));
    auto xbc_conv = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto x_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto B_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto C_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto y_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto gnormed  = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto gnorm_fp8= allocate_tensor<fp8>(TensorShape::make_2d(M, INTER));
    auto mixer    = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto xscale1  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto xscale2  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto amax     = allocate_tensor<unsigned int>(TensorShape::make_1d(1));
    auto ws       = allocate_tensor<char>(TensorShape::make_1d((int64_t)WS_BYTES));

    // 1. pre-norm RMSNorm (bf16)
    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    // 2. 量化 normed → fp8(per-tensor)，3 个 in_proj 行切片复用同一激活
    quantize_activation_fp8(normed.data_, norm_fp8.data_, xscale1.data_,
                            amax.data_, (size_t)M * HIDDEN, stream);
    gemm_fp8(norm_fp8.data_, w.in_proj_w,
             gate.data_, xscale1.data_, w.in_proj_wscale,
             M, INTER, HIDDEN, ws.data_, WS_BYTES, stream);
    gemm_fp8(norm_fp8.data_, w.in_proj_w + (size_t)INTER * HIDDEN,
             xbc.data_, xscale1.data_, w.in_proj_wscale + INTER,
             M, CONV_DIM, HIDDEN, ws.data_, WS_BYTES, stream);
    gemm_fp8(norm_fp8.data_, w.in_proj_w + (size_t)(INTER + CONV_DIM) * HIDDEN,
             dt.data_, xscale1.data_, w.in_proj_wscale + (INTER + CONV_DIM),
             M, H, HIDDEN, ws.data_, WS_BYTES, stream);

    // 3. depthwise causal conv1d + SiLU (bf16)
    causal_conv1d_prefill_bf16<64>(xbc.data_, w.conv1d_w, w.conv1d_b,
                                   xbc_conv.data_, conv_state_out, CONV_DIM, S, B, stream);

    // 4. split xBC → x | B | C (bf16)
    cudaMemcpy2DAsync(x_buf.data_, (size_t)INTER * BF,
                      xbc_conv.data_, (size_t)CONV_DIM * BF,
                      (size_t)INTER * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(B_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(C_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER + G * N, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);

    // 5. SSD scan (chunked TF32；cumsum/exp fp32 红线不破)
    ssd_scan_chunked_prefill_bf16<H, P, N, 128>(
        x_buf.data_, dt.data_, w.A_log, B_buf.data_, C_buf.data_,
        w.D, w.dt_bias, y_buf.data_, ssm_state_out,
        B, S, G, 0.f, FLT_MAX, stream);

    // 6. gated RMSNorm (bf16)
    rmsnorm_gated_bf16<G>(y_buf.data_, gnormed.data_, w.gnorm_w, gate.data_,
                          M, INTER, GROUP_SIZE, EPS, stream);

    // 7. out_proj: 量化 gnormed → fp8，gemm_fp8
    quantize_activation_fp8(gnormed.data_, gnorm_fp8.data_, xscale2.data_,
                            amax.data_, (size_t)M * INTER, stream);
    gemm_fp8(gnorm_fp8.data_, w.out_proj_w,
             mixer.data_, xscale2.data_, w.out_proj_wscale,
             M, HIDDEN, INTER, ws.data_, WS_BYTES, stream);

    // 8. 残差 (bf16)
    elementwise_ops_bf16<kElementwiseAdd>(
        input, mixer.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

// ===========================================================================
// FP8 decode：单步生成，in_proj/out_proj 走原生 fp8。
//   decode 是带宽受限：fp8 权重 1 字节（vs bf16 2 字节）→ 每 token 少读一半权重，
//   且与磁盘原生 fp8 存储一致（无需 dequant）。其余同 bf16 decode。
//   ⚠️ M=1 下 cuBLASLt fp8 matmul 退化为 gemv，tensor core 利用率低——收益主要来自
//      少读权重字节；实测若不及预期，可换自定义 fp8 gemv。
// ===========================================================================
template<int HIDDEN, int H, int P, int N, int G, int CONV_K = 4>
inline void mamba_block_decode_fp8(
    const __nv_bfloat16* input,          // [B, HIDDEN] 单 token
    const Mamba2BlockWeightsFP8& w,
    float* conv_state,                   // [B, CONV_DIM, CONV_K-1] in/out
    float* ssm_state,                    // [B, H, D, N] in/out
    __nv_bfloat16* out,                  // [B, HIDDEN]
    int B,
    cudaStream_t stream = nullptr
) {
    using namespace nemotron::ops;
    using namespace nemotron::ops::mamba2;
    using bf16 = __nv_bfloat16;
    using fp8  = __nv_fp8_e4m3;

    constexpr int INTER      = H * P;
    constexpr int CONV_DIM   = INTER + 2 * G * N;
    constexpr int GROUP_SIZE = INTER / G;
    const int M = B;
    constexpr float EPS = 1e-5f;
    constexpr size_t BF = sizeof(bf16);
    constexpr size_t WS_BYTES = 8ull * 1024 * 1024;   // cuBLASLt workspace（decode M=1，够用）

    auto normed   = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto norm_fp8 = allocate_tensor<fp8>(TensorShape::make_2d(M, HIDDEN));
    auto gate     = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto xbc      = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto dt       = allocate_tensor<bf16>(TensorShape::make_2d(M, H));
    auto xbc_conv = allocate_tensor<bf16>(TensorShape::make_2d(M, CONV_DIM));
    auto x_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto B_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto C_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, G * N));
    auto y_buf    = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto gnormed  = allocate_tensor<bf16>(TensorShape::make_2d(M, INTER));
    auto gnorm_fp8= allocate_tensor<fp8>(TensorShape::make_2d(M, INTER));
    auto mixer    = allocate_tensor<bf16>(TensorShape::make_2d(M, HIDDEN));
    auto xscale1  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto xscale2  = allocate_tensor<float>(TensorShape::make_1d(1));
    auto amax     = allocate_tensor<unsigned int>(TensorShape::make_1d(1));
    auto wsp      = allocate_tensor<char>(TensorShape::make_1d((int64_t)WS_BYTES));

    // 1. pre-norm RMSNorm
    rmsnorm_bf16(input, normed.data_, w.block_norm_w, M, HIDDEN, EPS, stream);

    // 2. 量化 normed → fp8，3 个 in_proj 行切片复用同一激活
    quantize_activation_fp8(normed.data_, norm_fp8.data_, xscale1.data_,
                            amax.data_, (size_t)M * HIDDEN, stream);
    gemm_fp8(norm_fp8.data_, w.in_proj_w,
             gate.data_, xscale1.data_, w.in_proj_wscale,
             M, INTER, HIDDEN, wsp.data_, WS_BYTES, stream);
    gemm_fp8(norm_fp8.data_, w.in_proj_w + (size_t)INTER * HIDDEN,
             xbc.data_, xscale1.data_, w.in_proj_wscale + INTER,
             M, CONV_DIM, HIDDEN, wsp.data_, WS_BYTES, stream);
    gemm_fp8(norm_fp8.data_, w.in_proj_w + (size_t)(INTER + CONV_DIM) * HIDDEN,
             dt.data_, xscale1.data_, w.in_proj_wscale + (INTER + CONV_DIM),
             M, H, HIDDEN, wsp.data_, WS_BYTES, stream);

    // 3. conv1d decode（更新 conv_state）
    causal_conv1d_decode_bf16<CONV_K>(xbc.data_, w.conv1d_w, w.conv1d_b,
                                      xbc_conv.data_, conv_state, CONV_DIM, B, stream);

    // 4. split x | B | C
    cudaMemcpy2DAsync(x_buf.data_, (size_t)INTER * BF,
                      xbc_conv.data_, (size_t)CONV_DIM * BF,
                      (size_t)INTER * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(B_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(C_buf.data_, (size_t)(G * N) * BF,
                      xbc_conv.data_ + INTER + G * N, (size_t)CONV_DIM * BF,
                      (size_t)(G * N) * BF, M, cudaMemcpyDeviceToDevice, stream);

    // 5. ssm decode 单步（更新 ssm_state）
    ssm_decode_bf16<H, P, N><<<dim3(B * H), 128, 0, stream>>>(
        x_buf.data_, dt.data_, w.A_log, B_buf.data_, C_buf.data_,
        w.D, ssm_state, y_buf.data_, w.dt_bias, B, G, 0.f, FLT_MAX);

    // 6. gated RMSNorm
    rmsnorm_gated_bf16<G>(y_buf.data_, gnormed.data_, w.gnorm_w, gate.data_,
                          M, INTER, GROUP_SIZE, EPS, stream);

    // 7. out_proj: 量化 gnormed → fp8，gemm_fp8
    quantize_activation_fp8(gnormed.data_, gnorm_fp8.data_, xscale2.data_,
                            amax.data_, (size_t)M * INTER, stream);
    gemm_fp8(gnorm_fp8.data_, w.out_proj_w,
             mixer.data_, xscale2.data_, w.out_proj_wscale,
             M, HIDDEN, INTER, wsp.data_, WS_BYTES, stream);

    // 8. 残差
    elementwise_ops_bf16<kElementwiseAdd>(
        input, mixer.data_, out, (size_t)M * HIDDEN,
        1.f, nullptr, 0.f, 1.f, stream);
}

}  // namespace nemotron::model

#endif  // NEMOTRON_INFER_MODEL_MAMBA_BLOCK_CUH

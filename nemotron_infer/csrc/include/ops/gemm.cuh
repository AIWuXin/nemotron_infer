//
// Created by Administrator on 2026/6/12.
//

#ifndef NEMOTRON_INFER_GEMM_CUH
#define NEMOTRON_INFER_GEMM_CUH

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace nemotron::ops {

// ===========================================================================
// cuBLAS handle 单例
// ===========================================================================
inline cublasHandle_t& cublas_handle() {
    static cublasHandle_t h = []{
        cublasHandle_t tmp{};
        cublasCreate(&tmp);
        return tmp;
    }();
    return h;
}

// ===========================================================================
// FP32 GEMM:  y[M,N] = x[M,K] @ W[N,K]^T
//
// cuBLAS 使用列主序。我们存的是行主序数据。
// 对于 y_row = x_row @ W^T_row:
//   等价于在列主序下计算 y_col = W_col @ x_col
//
// 参数:
//   M  = batch×seq  (输出行数)
//   N  = output_dim (输出列数)
//   K  = input_dim  (归约维度)
//   x  = 输入 [M, K], row-major, leading dim = K
//   W  = 权重 [N, K], row-major, leading dim = K
//   y  = 输出 [M, N], row-major, leading dim = N
// ===========================================================================
inline void gemm_fp32(
    const float* x,
    const float* W,
    float* y,
    const int M,
    const int N,
    const int K,
    bool transpose_W = true,
    cudaStream_t stream = nullptr
) {
    const float alpha = 1.f, beta = 0.f;
    cublasSetStream(cublas_handle(), stream);

    if (transpose_W) {
        // y[M,N] = x[M,K] @ W[N,K]^T
        // 列主序: y'[N,M] = W'[N,K] @ x'[K,M]
        //   op(W') = N  → W' [N,K] 列主序
        //   op(x') = N  → x' [K,M] 列主序
        //   结果: [N,K] @ [K,M] = [N,M] 列主序 = y_row [M,N]
        cublasSgemm(cublas_handle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    W, K,      // W[N,K] row-major → cuBLAS sees as [K,N] col-major. OP_T → [N,K] col-major. lda = K.
                    x, K,      // x[M,K] row-major → cuBLAS sees as [K,M] col-major. OP_N → [K,M] col-major. ldb = K.
                    &beta,
                    y, N);     // y[M,N] row-major → cuBLAS sees as [N,M] col-major. ldc = N.
    } else {
        // y[M,N] = x[M,K] @ W[K,N]  (W already transposed in memory)
        cublasSgemm(cublas_handle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    W, N,      // W[K,N] row-major → cuBLAS sees as [N,K] col-major. lda = N.
                    x, K,      // x[M,K] row-major → cuBLAS sees as [K,M] col-major. ldb = K.
                    &beta,
                    y, N);
    }
}

// ===========================================================================
// BF16 GEMM:  y[M,N] = x[M,K] @ W[N,K]^T
// 内部使用 TF32 Tensor Core 计算 (CUBLAS_COMPUTE_32F)
// 数据 IO 为 BF16
// ===========================================================================
inline void gemm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* y,
    const int M,
    const int N,
    const int K,
    cudaStream_t stream = nullptr
) {
    const float alpha = 1.f, beta = 0.f;
    cublasSetStream(cublas_handle(), stream);

    cublasGemmEx(cublas_handle(),
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 W, CUDA_R_16BF, K,
                 x, CUDA_R_16BF, K,
                 &beta,
                 y, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F,        // 累加器 FP32（TF32 Tensor Core）
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


// ===========================================================================
// FP8 GEMM (cuBLASLt):  y[M,N] = x[M,K] @ W_fp8[N,K]^T
//
// cuBLASLt 内部自动将 BF16 activation 量化为 FP8（per-tensor scale）
// 权重为预处理的 FP8 E4M3（per-column scale）
// 输出为 BF16
//
// 调用前需要预处理权重：
//   fp8_preprocess_weight(w_fp8_raw, w_scale_group16, N, K, w_fp8_col, w_scale_col);
// ===========================================================================

// ===========================================================================
// cuBLASLt handle 单例
// ===========================================================================
inline cublasLtHandle_t& cublaslt_handle() {
    static cublasLtHandle_t h = []{
        cublasLtHandle_t tmp{};
        cublasLtCreate(&tmp);
        return tmp;
    }();
    return h;
}

// ===========================================================================
// 权重预处理: per-16-group scale → per-row(输出通道 N) scale
//
// w_raw   [N, K]: 原始 FP8 权重 (E4M3), 包含 per-16-group 的 scale
// w_scale [N, K/16]: 每 16 个元素的 scale factor
// → 输出 w_out [N, K]: 重新量化的 FP8 权重, per-row scale
// → 输出 s_out [N]: 每个输出通道(行)的 scale factor
//
// ⚠️ scale 必须沿输出维 N（行），不能沿归约维 K。
//    沿 K 的 scale 落在 matmul 的 sum_k 内部、随 k 变化，
//    任何单次 FP8 tensor-core matmul 都无法将其提取出来。
//    沿 N 的 scale 可在 matmul 之后由 epilogue 按列分离。
// ===========================================================================
inline void fp8_preprocess_weight(
    const __nv_fp8_e4m3* w_raw,
    const float* w_scale,
    __nv_fp8_e4m3* w_out,
    float* s_out,
    const int N,
    const int K
) {
    // 对每行(输出通道)求最大绝对值 → 行 scale
    for (int n = 0; n < N; ++n) {
        float row_max = 0.f;
        for (int k = 0; k < K; ++k) {
            int group = k / 16;
            float val = static_cast<float>(w_raw[n * K + k]) * w_scale[n * (K/16) + group];
            row_max = fmaxf(row_max, fabsf(val));
        }
        s_out[n] = row_max / 448.f;
        if (s_out[n] < 1e-10f) s_out[n] = 1.f / 448.f;

        // 重新量化
        float inv = 1.f / s_out[n];
        for (int k = 0; k < K; ++k) {
            int group = k / 16;
            float val = static_cast<float>(w_raw[n * K + k]) * w_scale[n * (K/16) + group];
            float quant = val * inv;
            quant = fmaxf(-448.f, fminf(448.f, quant));
            w_out[n * K + k] = static_cast<__nv_fp8_e4m3>(quant);
        }
    }
}

// ===========================================================================
// FP8 epilogue: 按输出通道 n 应用 per-row 权重 scale
//   y[m, n] *= w_scale[n]
// matmul 已乘过 activation 的 per-tensor scale，此处只补权重列 scale。
// ===========================================================================
__global__ void fp8_apply_col_scale_kernel(
    __nv_bfloat16* __restrict__ y,
    const float* __restrict__ w_scale,   // [N]
    const int M,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int n = idx % N;
    float v = __bfloat162float(y[idx]) * w_scale[n];
    y[idx] = __float2bfloat16_rn(v);
}

// ===========================================================================
// FP8 GEMM: y[M,N] = (x[M,K] * a_scale) @ (W[N,K]^T * w_scale)
//
// 三步走：
//   ① 加载时：fp8_preprocess_weight → W_fp8 + per-column scale
//   ② 运行时：量化 BF16 activation → FP8 + per-tensor scale
//   ③ cuBLASLt FP8 matmul → BF16 output
//
// E4M3 格式表示范围 [-448, 448]（整数），实际值 = quant_int × scale
// ===========================================================================

// ===========================================================================
// 激活量化: BF16 → FP8 E4M3 (per-tensor)
// 返回 scale factor，调用者负责为 x_fp8 分配 M*K 空间
// ===========================================================================
inline float quantize_activation_fp8(
    const __nv_bfloat16* x_bf16,
    __nv_fp8_e4m3* x_fp8,
    const int M,
    const int K,
    cudaStream_t stream = nullptr
) {
    // TODO: 用 kernel 在 GPU 上求 max_abs + 量化，这里先给伪代码
    // 实际需要 launch 一个 reduction kernel
    //
    // float amax = block_reduce_max(abs(x_bf16[i]))
    // float a_scale = amax / 448.f
    // x_fp8[i] = static_cast<__nv_fp8_e4m3>(__bfloat162float(x_bf16[i]) / a_scale)
    (void)x_bf16; (void)x_fp8; (void)M; (void)K; (void)stream;
    return 1.f / 448.f;  // placeholder
}

// ===========================================================================
// FP8 GEMM 核心调用
// x_fp8:     量化后的 activation [M, K]
// x_scale:   激活 per-tensor scale（device 指针，单 float）
// W_fp8:     预处理后的 weight [N, K]
// w_scale:   权重 per-row scale [N]（device 指针，沿输出通道 N）
// y_bf16:    BF16 output [M, N]
// workspace: 调用方预分配的 cuBLASLt workspace（device 指针），可为 nullptr
// ws_size:   workspace 字节数
//
// ⚠️ A_SCALE_POINTER 必须是 device 内存地址（per-tensor 标量）
// ⚠️ 权重 scale 沿输出维 N，无法作为 B_SCALE 喂给 matmul（那是 per-tensor 标量
//    语义），改为 matmul 后用 epilogue kernel 按列 n 乘 w_scale[n]。
// ⚠️ workspace 由调用方持有，算子内零 cudaMalloc/cudaFree（cudaFree 隐式同步，
//    放热路径里会阻塞流水线，是 FP8 慢于 BF16 的主因）。
// ===========================================================================
inline void gemm_fp8(
    const __nv_fp8_e4m3* x_fp8,
    const __nv_fp8_e4m3* W_fp8,
    __nv_bfloat16* y_bf16,
    const float* x_scale,        // device ptr to per-tensor activation scale
    const float* w_scale,        // device ptr to per-row weight scale [N]
    const int M,
    const int N,
    const int K,
    void* workspace = nullptr,   // 调用方预分配，引擎上下文持有
    size_t ws_size = 0,
    cudaStream_t stream = nullptr
) {
    const auto order = CUBLASLT_ORDER_ROW;

    // A: activation x [M, K] row-major (FP8)
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, K);
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // B: weight W [N, K] row-major (FP8), will be transposed
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, N, K, K);
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // C/D: output y [M, N] row-major (BF16)
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, N);
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // Matmul: D = A @ B^T
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    const auto trans_b = CUBLASLT_MATMUL_DESC_TRANSB;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));

    // Activation scale (per-tensor, device memory)
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                   &x_scale, sizeof(x_scale));

    // 权重 scale 沿输出维 N，无法用 B_SCALE_POINTER（per-tensor 标量语义）表达，
    // 在 matmul 之后由 epilogue kernel 按列施加。

    // Heuristic
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws_size, sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle(), opDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                   pref, 1, &heuristic, &returned);

    // Execute — workspace 由调用方提供，算子内零分配
    const float alpha = 1.f, beta = 0.f;
    cublasLtMatmul(cublaslt_handle(), opDesc,
                   &alpha,
                   x_fp8, Adesc,
                   W_fp8, Bdesc,
                   &beta,
                   y_bf16, Cdesc,
                   y_bf16, Cdesc,
                   &heuristic.algo,
                   workspace, ws_size, stream);

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtMatmulPreferenceDestroy(pref);

    // epilogue: 施加 per-row(输出通道 N) 权重 scale
    const int total = M * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    fp8_apply_col_scale_kernel<<<blocks, threads, 0, stream>>>(y_bf16, w_scale, M, N);
}

}  // namespace nemotron::ops

#endif //NEMOTRON_INFER_GEMM_CUH

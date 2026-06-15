//
// sdpa_cudnn.h — cuDNN frontend SDPA 封装的「声明」
//
// ⚠️ 实现放在 sdpa_cudnn.cpp（纯 host TU，由 MSVC 编译）。
//    cudnn_frontend 是纯 host C++ header-only 库，含巨型 R"KERNEL(...)"
//    原始字符串字面量，NVCC 的 cudafe++ 前端解析时会崩溃
//    （internal error in literals.c）。因此 frontend 头绝不能进 .cu。
//    本头只暴露一个普通函数声明，.cu 侧可安全 include。
//
#ifndef NEMOTRON_INFER_SDPA_CUDNN_H
#define NEMOTRON_INFER_SDPA_CUDNN_H

#ifdef USE_CUDNN

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nemotron::ops::attention {

// 返回 true 表示 cuDNN 路径成功执行；false 表示需回退到手写 kernel。
// 数据布局：Q/K/V/O 均为 [H, S, head_dim] 连续，单 batch(B=1)，causal。
bool sdpa_prefill_bf16_cudnn(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, int S, int H, int head_dim, cudaStream_t stream);

}  // namespace nemotron::ops::attention

#endif  // USE_CUDNN
#endif  // NEMOTRON_INFER_SDPA_CUDNN_H

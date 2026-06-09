#pragma once
// ===========================================================================
// cudnn_ctx.h  —  全局 cuDNN 上下文（进程级单例）
//
// 所有算子共享同一个 handle 和 tensor descriptor，避免重复创建开销。
// 使用方法：
//   auto& ctx = nemotron::CudnnContext::instance();
//   cudnnSetStream(ctx.handle(), stream);
// ===========================================================================

#include <cuda_runtime.h>
#include <cudnn.h>

namespace nemotron {

class CudnnContext {
public:
    static CudnnContext& instance() {
        static CudnnContext ctx;
        return ctx;
    }

    cudnnHandle_t handle() const { return handle_; }
    cudnnTensorDescriptor_t tensor_desc() const { return tensor_desc_; }

private:
    CudnnContext() {
        cudnnCreate(&handle_);
        cudnnCreateTensorDescriptor(&tensor_desc_);
    }
    ~CudnnContext() {
        cudnnDestroyTensorDescriptor(tensor_desc_);
        cudnnDestroy(handle_);
    }
    CudnnContext(const CudnnContext&) = delete;
    CudnnContext& operator=(const CudnnContext&) = delete;

    cudnnHandle_t handle_{};
    cudnnTensorDescriptor_t tensor_desc_{};
};

}  // namespace nemotron

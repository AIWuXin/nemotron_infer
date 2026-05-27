#pragma once
// ===========================================================================
// tensor.h  —  轻量推理专用张量（4D, GPU, 128-byte 对齐）
//
// 设计原则：
//  1. 纯 POD 结构体，无虚函数，可安全在 host/device 之间传递
//  2. 零开销 view：reshape/slice 只改 shape/strides，不拷贝数据
//  3. 128-byte 对齐内存：适配 GPU cache line & 向量化访存
//  4. 显式 owner 标记：避免推断场景无意义的内存管理开销
//  5. 异步操作：所有数据搬运通过 cudaStream_t 接口
// ===========================================================================

#include <cstdint>
#include <cuda_fp16.h>    // __half, __nv_bfloat16
#include <cuda_bf16.h>
#include <cuda_fp8.h>     // __nv_fp8_e4m3

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// ===========================================================================
// 1. 基础类型别名
// ===========================================================================
using float16_t = __half;
using bfloat16_t = __nv_bfloat16;
using float8_t  = __nv_fp8_storage_t;   // FP8 E4M3 (权重存储), CUDA 12.x 用 storage 类型
using float32_t = float;

namespace nemotron {

// ===========================================================================
// 2. TensorShape  —  编译期形状描述（帮助编译器优化循环）
// ===========================================================================
struct TensorShape {
    static constexpr int MAX_DIM = 4;
    int64_t dims[MAX_DIM];    // [batch, heads, seq_len, feat_dim]
    int64_t strides[MAX_DIM]; // stride[i] = ∏_{j>i} dims[j], row-major
    int     ndim = 0;

    CUDA_HOST_DEVICE int64_t size() const {
        if (ndim == 0) return 0;
        int64_t s = 1;
        for (int i = 0; i < ndim; ++i) s *= dims[i];
        return s;
    }

    // ---- 便于构造 shape 的工厂方法 ----
    static TensorShape make_1d(int64_t d0) {
        TensorShape s;
        s.dims[0] = d0;  s.strides[0] = 1;
        s.ndim = 1;
        return s;
    }
    static TensorShape make_2d(int64_t d0, int64_t d1) {
        TensorShape s;
        s.dims[0] = d0;  s.strides[0] = d1;
        s.dims[1] = d1;  s.strides[1] = 1;
        s.ndim = 2;
        return s;
    }
    static TensorShape make_3d(int64_t d0, int64_t d1, int64_t d2) {
        TensorShape s;
        s.dims[0] = d0;  s.strides[0] = d1 * d2;
        s.dims[1] = d1;  s.strides[1] = d2;
        s.dims[2] = d2;  s.strides[2] = 1;
        s.ndim = 3;
        return s;
    }
    static TensorShape make_4d(int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
        TensorShape s;
        int64_t stride = d1 * d2 * d3;
        s.dims[0] = d0;  s.strides[0] = stride;
        stride /= d1;
        s.dims[1] = d1;  s.strides[1] = stride;
        stride /= d2;
        s.dims[2] = d2;  s.strides[2] = stride;
        s.dims[3] = d3;  s.strides[3] = 1;
        s.ndim = 4;
        return s;
    }
};

// ===========================================================================
// 3. Tensor<T>  —  设备张量（POD 结构体）
//
// 注意：
//  - data_  始终指向 GPU 内存（device pointer）
//  - owner_ 为 true 时，析构/移动赋值会释放 GPU 内存
//  - view() 创建的张量 owner_=false，不管理内存
// ===========================================================================
template <typename T>
struct Tensor {
    T*           data_   = nullptr;
    TensorShape  shape_;
    bool         owner_  = false;   // 是否持有 GPU 内存所有权

    // ---- 默认构造 ----
    Tensor() = default;

    // ---- 禁止拷贝（GPU 内存不应被浅拷贝）----
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // ---- 移动构造 ----
    Tensor(Tensor&& other) noexcept
        : data_(other.data_), shape_(other.shape_), owner_(other.owner_) {
        other.data_  = nullptr;
        other.owner_ = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owner_ && data_) { /* cudaFree(data_); */ } // 由 allocator 管理
            data_  = other.data_;
            shape_ = other.shape_;
            owner_ = other.owner_;
            other.data_  = nullptr;
            other.owner_ = false;
        }
        return *this;
    }

    // ---- 便利访问 ----
    CUDA_HOST_DEVICE int64_t size()       const { return shape_.size(); }
    CUDA_HOST_DEVICE int64_t bytes()      const { return size() * sizeof(T); }
    CUDA_HOST_DEVICE int     ndim()       const { return shape_.ndim; }
    CUDA_HOST_DEVICE int64_t dim(int i)   const { return shape_.dims[i]; }
    CUDA_HOST_DEVICE int64_t stride(int i)const { return shape_.strides[i]; }

    // ---- 判断 ----
    bool empty()  const { return data_ == nullptr || size() == 0; }
    bool owner()  const { return owner_; }

    // =======================================================================
    // View 操作（零拷贝，返回非 owning Tensor）
    // =======================================================================

    /// 按 batch 索引切片  [b, :, :, :]
    CUDA_HOST_DEVICE Tensor<T> view_batch(int64_t b) const {
        Tensor<T> view;
        view.data_  = data_ + b * stride(0);
        view.shape_ = shape_;
        view.shape_.ndim -= 1;
        for (int i = 0; i < view.shape_.ndim; ++i) {
            view.shape_.dims[i]    = shape_.dims[i+1];
            view.shape_.strides[i] = shape_.strides[i+1];
        }
        view.owner_ = false;
        return view;
    }

    /// 按第 dim 维切片（最常用: dim=1 选 head）
    CUDA_HOST_DEVICE Tensor<T> view_slice(int dim_idx, int64_t index) const {
        Tensor<T> view;
        view.data_  = data_ + index * stride(dim_idx);
        view.shape_ = shape_;
        // 移除 dim_idx 维
        for (int i = dim_idx; i < view.shape_.ndim - 1; ++i) {
            view.shape_.dims[i]    = shape_.dims[i+1];
            view.shape_.strides[i] = shape_.strides[i+1];
        }
        view.shape_.ndim -= 1;
        view.owner_ = false;
        return view;
    }

    /// 将前两维合并（[B,S,D] -> [B*S,D]），用于 gemm 批量处理
    Tensor<T> view_2d(int64_t d0, int64_t d1) const {
        Tensor<T> view;
        view.data_  = data_;
        view.shape_ = TensorShape::make_2d(d0, d1);
        view.owner_ = false;
        return view;
    }

    /// 返回相同内存的新形状视图
    Tensor<T> view_reshape(const TensorShape& new_shape) const {
        Tensor<T> view;
        view.data_  = data_;
        view.shape_ = new_shape;
        view.owner_ = false;
        return view;
    }
};

// ===========================================================================
// 4. 设备工具函数（在 allocator.cuh 中实现）
// ===========================================================================

/// 分配 GPU 内存（128-byte 对齐），返回 owning Tensor
/// 参数 alignment 指定对齐字节数
template <typename T>
Tensor<T> allocate_tensor(const TensorShape& shape, int alignment = 128);

/// 分配并初始化为零
template <typename T>
Tensor<T> allocate_tensor_zeros(const TensorShape& shape, int alignment = 128);

/// 释放 owning tensor 持有的 GPU 内存
template <typename T>
void free_tensor(Tensor<T>& tensor);

/// 从 host 数据拷贝到 owning tensor（异步）
template <typename T>
void copy_host_to_device(Tensor<T>& dst, const T* src, cudaStream_t stream = 0);

/// 从 device 数据拷贝到 host buffer（异步）
template <typename T>
void copy_device_to_host(T* dst, const Tensor<T>& src, cudaStream_t stream = 0);

/// 设备间拷贝（dst 必须已分配且 size 匹配）
template <typename T>
void copy_device_to_device(Tensor<T>& dst, const Tensor<T>& src, cudaStream_t stream = 0);

/// 将 tensor 置零（异步）
template <typename T>
void device_memset_zero(Tensor<T>& tensor, cudaStream_t stream = 0);

} // namespace nemotron

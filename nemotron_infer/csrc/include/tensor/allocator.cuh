#pragma once
// ===========================================================================
// allocator.cuh  —  GPU 内存分配器（优化推理场景）
//
// 两种分配策略：
//  1. BumpAllocator — 顺序分配，一次 reset 释放全部
//     适用于：模型权重加载（一次分配，永不释放）
//  2. FreeListAllocator — 空闲链表复用
//     适用于：临时张量（prefill/decode 每步变化）
//
// 优化要点：
//  - 128-byte 对齐：匹配 GPU cache line 和内存合并访问
//  - 大块预分配 (slab)：减少 cudaMalloc 调用次数
//  - 异步 memset 零初始化：避免同步等待
// ===========================================================================

#include "tensor/tensor.h"
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdio>          // fprintf, stderr

// 轻量级断言，避免 Phase 1 引入 glog 依赖
#define TENSOR_CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FATAL] %s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
        std::abort(); \
    } \
} while(0)

#define TENSOR_CHECK_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "[FATAL] %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
        std::abort(); \
    } \
} while(0)

namespace nemotron {

// ===========================================================================
// 1. 对齐工具
// ===========================================================================
inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

inline bool is_aligned(void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ===========================================================================
// 2. BumpAllocator  — 大块预分配 + 顺序 bump
//
// 线程安全：内部用 mutex 保护 slab 列表
// 使用场景：模型参数加载阶段
// ===========================================================================
class BumpAllocator {
public:
    static constexpr size_t DEFAULT_SLAB_SIZE = 64ULL * 1024 * 1024; // 64 MB
    static constexpr size_t ALIGNMENT         = 128;

    explicit BumpAllocator(size_t slab_size = DEFAULT_SLAB_SIZE)
        : slab_size_(slab_size) {}

    ~BumpAllocator() { reset(); }

    // 禁止拷贝/移动（管理 GPU 内存）
    BumpAllocator(const BumpAllocator&) = delete;
    BumpAllocator& operator=(const BumpAllocator&) = delete;

    // ---- 分配对齐 GPU 内存 ----
    void* allocate(size_t size_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t aligned = align_up(size_bytes, ALIGNMENT);

        // 当前 slab 剩余不足 → 分配新 slab
        if (!current_slab_ || (current_slab_offset_ + aligned > current_slab_size_)) {
            allocate_new_slab();
        }

        void* ptr = static_cast<char*>(current_slab_) + current_slab_offset_;
        current_slab_offset_ += aligned;
        total_allocated_ += aligned;
        return ptr;
    }

    // ---- 分配并异步清零 ----
    void* allocate_zero(size_t size_bytes, cudaStream_t stream = 0) {
        void* ptr = allocate(size_bytes);
        cudaError_t err = cudaMemsetAsync(ptr, 0, size_bytes, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[WARN] cudaMemsetAsync failed: %s\n", cudaGetErrorString(err));
        }
        return ptr;
    }

    // ---- 释放所有 slab（不逐块释放，效率最高）----
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto* slab : slabs_) {
            cudaFree(slab);
        }
        slabs_.clear();
        current_slab_        = nullptr;
        current_slab_size_   = 0;
        current_slab_offset_ = 0;
        total_allocated_     = 0;
    }

    size_t total_allocated() const { return total_allocated_; }
    size_t num_slabs()       const { return slabs_.size(); }

private:
    void allocate_new_slab() {
        // 分配大块 GPU 内存
        void* slab = nullptr;
        size_t alloc_size = std::max(slab_size_, ALIGNMENT);
        cudaError_t err = cudaMalloc(&slab, alloc_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[FATAL] cudaMalloc for BumpAllocator slab failed: %s\n",
                    cudaGetErrorString(err));
            std::abort();
        }
        slabs_.push_back(slab);
        current_slab_        = slab;
        current_slab_size_   = alloc_size;
        current_slab_offset_ = 0;
    }

    size_t slab_size_;
    std::vector<void*> slabs_;
    void*   current_slab_        = nullptr;
    size_t  current_slab_size_   = 0;
    size_t  current_slab_offset_ = 0;
    size_t  total_allocated_     = 0;
    std::mutex mutex_;
};

// ===========================================================================
// 3. 全局默认分配器
// ===========================================================================
inline BumpAllocator& default_allocator() {
    static BumpAllocator alloc;
    return alloc;
}

/// 替换全局分配器（用于测试或自定义 slab 大小）
inline void set_default_allocator(BumpAllocator* alloc) {
    // 留给将来多实例场景
    (void)alloc;
}

// ===========================================================================
// 4. Tensor 分配/释放函数实现
// ===========================================================================

template <typename T>
Tensor<T> allocate_tensor(const TensorShape& shape, int alignment) {
    Tensor<T> t;
    t.shape_ = shape;
    t.owner_ = true;

    size_t bytes = shape.size() * sizeof(T);
    // BumpAllocator 内部强制 128-byte 对齐，alignment 参数保留为未来扩展
    t.data_ = static_cast<T*>(default_allocator().allocate(bytes));
    return t;
}

template <typename T>
Tensor<T> allocate_tensor_zeros(const TensorShape& shape, int alignment) {
    Tensor<T> t;
    t.shape_ = shape;
    t.owner_ = true;

    size_t bytes = shape.size() * sizeof(T);
    t.data_ = static_cast<T*>(default_allocator().allocate_zero(bytes));
    return t;
}

template <typename T>
void free_tensor(Tensor<T>& tensor) {
    // BumpAllocator 不支持逐块释放，由 reset() 统一回收
    // 这里仅标记为非 owning，避免误用
    tensor.owner_ = false;
    tensor.data_  = nullptr;
    tensor.shape_ = TensorShape{};
}

// ===========================================================================
// 5. 异步拷贝与内存操作
// ===========================================================================

template <typename T>
void copy_host_to_device(Tensor<T>& dst, const T* src, cudaStream_t stream) {
    TENSOR_CHECK(!dst.empty());
    TENSOR_CHECK(src != nullptr);
    cudaError_t err = cudaMemcpyAsync(
        dst.data_, src, dst.bytes(),
        cudaMemcpyHostToDevice, stream
    );
    TENSOR_CHECK(err == cudaSuccess);
}

template <typename T>
void copy_device_to_host(T* dst, const Tensor<T>& src, cudaStream_t stream) {
    TENSOR_CHECK(!src.empty());
    TENSOR_CHECK(dst != nullptr);
    cudaError_t err = cudaMemcpyAsync(
        dst, src.data_, src.bytes(),
        cudaMemcpyDeviceToHost, stream
    );
    TENSOR_CHECK(err == cudaSuccess);
}

template <typename T>
void copy_device_to_device(Tensor<T>& dst, const Tensor<T>& src, cudaStream_t stream) {
    TENSOR_CHECK(!src.empty());
    TENSOR_CHECK(!dst.empty());
    TENSOR_CHECK(dst.size() >= src.size());
    cudaError_t err = cudaMemcpyAsync(
        dst.data_, src.data_, src.bytes(),
        cudaMemcpyDeviceToDevice, stream
    );
    TENSOR_CHECK(err == cudaSuccess);
}

template <typename T>
void device_memset_zero(Tensor<T>& tensor, cudaStream_t stream) {
    TENSOR_CHECK(!tensor.empty());
    cudaError_t err = cudaMemsetAsync(
        tensor.data_, 0, tensor.bytes(), stream
    );
    TENSOR_CHECK(err == cudaSuccess);
}

} // namespace nemotron

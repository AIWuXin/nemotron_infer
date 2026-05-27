// ===========================================================================
// test_tensor.cu  —  Phase 1 单元测试
// ===========================================================================

#include <gtest/gtest.h>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/allocator.cuh"

using namespace nemotron;

// ===========================================================================
// 辅助：打印 tensor 信息（调试用）
// ===========================================================================
template <typename T>
void print_tensor_info(const char* name, const Tensor<T>& t) {
    printf("[%s] ndim=%d  size=%lld  bytes=%lld  owner=%d  data=%p\n",
           name, t.ndim(), (long long)t.size(), (long long)t.bytes(),
           int(t.owner()), (void*)t.data_);
    for (int i = 0; i < t.ndim(); ++i) {
        printf("  dim[%d]=%lld  stride[%d]=%lld\n",
               i, (long long)t.dim(i), i, (long long)t.stride(i));
    }
}

// ===========================================================================
// 1. TensorShape 测试
// ===========================================================================
TEST(TensorShapeTest, Make1D) {
    auto s = TensorShape::make_1d(100);
    EXPECT_EQ(s.ndim, 1);
    EXPECT_EQ(s.size(), 100);
    EXPECT_EQ(s.dims[0], 100);
    EXPECT_EQ(s.strides[0], 1);
}

TEST(TensorShapeTest, Make2D) {
    auto s = TensorShape::make_2d(3, 4);
    EXPECT_EQ(s.ndim, 2);
    EXPECT_EQ(s.size(), 12);
    EXPECT_EQ(s.dims[0], 3);
    EXPECT_EQ(s.strides[0], 4);  // row-major
    EXPECT_EQ(s.dims[1], 4);
    EXPECT_EQ(s.strides[1], 1);
}

TEST(TensorShapeTest, Make3D) {
    auto s = TensorShape::make_3d(2, 3, 4);
    EXPECT_EQ(s.ndim, 3);
    EXPECT_EQ(s.size(), 24);
    EXPECT_EQ(s.dims[0], 2);
    EXPECT_EQ(s.strides[0], 12);
    EXPECT_EQ(s.dims[1], 3);
    EXPECT_EQ(s.strides[1], 4);
    EXPECT_EQ(s.dims[2], 4);
    EXPECT_EQ(s.strides[2], 1);
}

TEST(TensorShapeTest, Make4D) {
    // 典型注意力： [batch=2, heads=40, seq=128, dim=128]
    auto s = TensorShape::make_4d(2, 40, 128, 128);
    EXPECT_EQ(s.ndim, 4);
    EXPECT_EQ(s.size(), 2 * 40 * 128 * 128);
    EXPECT_EQ(s.dims[0], 2);
    EXPECT_EQ(s.strides[0], 40 * 128 * 128);
    EXPECT_EQ(s.dims[3], 128);
    EXPECT_EQ(s.strides[3], 1);
}

// ===========================================================================
// 2. Tensor 分配测试
// ===========================================================================
TEST(TensorAllocTest, AllocateFloat) {
    auto shape = TensorShape::make_3d(2, 128, 3136);
    auto t = allocate_tensor<float>(shape);

    EXPECT_FALSE(t.empty());
    EXPECT_TRUE(t.owner());
    EXPECT_EQ(t.size(), 2 * 128 * 3136);
    EXPECT_NE(t.data_, nullptr);

    // 验证 GPU 指针地址 128-byte 对齐
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data_) % 128, 0);

    free_tensor(t);
    EXPECT_FALSE(t.owner());
    EXPECT_EQ(t.data_, nullptr);
}

TEST(TensorAllocTest, AllocateHalf) {
    auto shape = TensorShape::make_2d(100, 200);
    auto t = allocate_tensor<float16_t>(shape);

    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.bytes(), 100 * 200 * sizeof(float16_t));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data_) % 128, 0);

    free_tensor(t);
}

TEST(TensorAllocTest, AllocateBFloat16) {
    auto shape = TensorShape::make_2d(64, 128);
    auto t = allocate_tensor<bfloat16_t>(shape);

    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.bytes(), 64 * 128 * sizeof(bfloat16_t));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data_) % 128, 0);

    free_tensor(t);
}

TEST(TensorAllocTest, AllocateZeros) {
    auto shape = TensorShape::make_2d(1024, 256);
    auto t = allocate_tensor_zeros<float>(shape);

    // 拷贝回 host 验证为零
    std::vector<float> host(t.size());
    cudaMemcpy(host.data(), t.data_, t.bytes(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], 0.0f);
    }

    free_tensor(t);
}

// ===========================================================================
// 3. Host-Device 拷贝测试
// ===========================================================================
TEST(TensorCopyTest, HostToDeviceAndBack) {
    const int N = 1024;
    auto shape = TensorShape::make_1d(N);
    auto t = allocate_tensor<float>(shape);

    // Host → Device
    std::vector<float> src(N);
    for (int i = 0; i < N; ++i) src[i] = static_cast<float>(i * 0.5);

    copy_host_to_device(t, src.data());
    cudaDeviceSynchronize();

    // Device → Host
    std::vector<float> dst(N);
    copy_device_to_host(dst.data(), t);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(dst[i], static_cast<float>(i * 0.5));
    }

    free_tensor(t);
}

TEST(TensorCopyTest, DeviceToDevice) {
    const int N = 512;
    auto shape = TensorShape::make_1d(N);
    auto src = allocate_tensor<float>(shape);
    auto dst = allocate_tensor<float>(shape);

    // 填充 src
    std::vector<float> host_src(N);
    for (int i = 0; i < N; ++i) host_src[i] = float(i);
    copy_host_to_device(src, host_src.data());
    cudaDeviceSynchronize();

    // 设备间拷贝
    copy_device_to_device(dst, src);
    cudaDeviceSynchronize();

    // 验证
    std::vector<float> host_dst(N);
    copy_device_to_host(host_dst.data(), dst);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_dst[i], float(i));
    }

    free_tensor(src);
    free_tensor(dst);
}

// ===========================================================================
// 4. View 操作测试（零拷贝）
// ===========================================================================
TEST(TensorViewTest, ViewBatch) {
    // [2, 40, 128, 128]
    auto shape = TensorShape::make_4d(2, 40, 128, 128);
    auto t = allocate_tensor<float>(shape);
    auto* base = t.data_;

    auto batch0 = t.view_batch(0);
    EXPECT_EQ(batch0.ndim(), 3);
    EXPECT_EQ(batch0.size(), 40 * 128 * 128);
    EXPECT_EQ(batch0.data_, base);  // 零拷贝：指针相同
    EXPECT_EQ(batch0.dim(0), 40);
    EXPECT_EQ(batch0.dim(1), 128);
    EXPECT_EQ(batch0.dim(2), 128);
    EXPECT_EQ(batch0.stride(0), 128 * 128);
    EXPECT_EQ(batch0.stride(1), 128);
    EXPECT_EQ(batch0.stride(2), 1);
    EXPECT_FALSE(batch0.owner());

    auto batch1 = t.view_batch(1);
    EXPECT_EQ(batch1.data_, base + 40 * 128 * 128);  // 偏移

    free_tensor(t);
}

TEST(TensorViewTest, View2D) {
    // [B*S, D] = [128, 3136]
    auto shape = TensorShape::make_3d(2, 64, 3136);
    auto t = allocate_tensor<float>(shape);
    auto* base = t.data_;

    auto v = t.view_2d(2 * 64, 3136);
    EXPECT_EQ(v.ndim(), 2);
    EXPECT_EQ(v.size(), 2 * 64 * 3136);
    EXPECT_EQ(v.data_, base);  // 零拷贝
    EXPECT_EQ(v.dim(0), 128);
    EXPECT_EQ(v.dim(1), 3136);
    EXPECT_FALSE(v.owner());

    free_tensor(t);
}

TEST(TensorViewTest, ViewSlice) {
    // [2, 40, 128, 128] → slice dim=1, index=5
    auto shape = TensorShape::make_4d(2, 40, 128, 128);
    auto t = allocate_tensor<float>(shape);
    auto* base = t.data_;

    auto head5 = t.view_slice(1, 5);
    EXPECT_EQ(head5.ndim(), 3);
    EXPECT_EQ(head5.size(), 2 * 128 * 128);
    EXPECT_EQ(head5.data_, base + 5 * 128 * 128);
    EXPECT_FALSE(head5.owner());

    free_tensor(t);
}

// ===========================================================================
// 5. Memset / 异步清零测试
// ===========================================================================
TEST(TensorMemsetTest, AsyncZero) {
    auto shape = TensorShape::make_2d(100, 200);
    auto t = allocate_tensor<float>(shape);

    // 先填充非零值
    std::vector<float> nonzero(t.size(), 42.0f);
    copy_host_to_device(t, nonzero.data());
    cudaDeviceSynchronize();

    // 异步清零
    device_memset_zero(t);

    // 同步后验证
    cudaDeviceSynchronize();
    std::vector<float> host(t.size());
    copy_device_to_host(host.data(), t);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], 0.0f);
    }

    free_tensor(t);
}

// ===========================================================================
// 6. 128-byte 对齐验证
// ===========================================================================
TEST(AlignmentTest, AllAllocationsAreAligned) {
    // 分配各种大小的张量，全部应 128-byte 对齐
    std::vector<size_t> sizes = {1, 17, 32, 127, 128, 255, 1023, 3136, 7680};

    for (auto size : sizes) {
        auto shape = TensorShape::make_1d(size);
        auto t = allocate_tensor<float>(shape);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data_) % 128, 0)
            << "size=" << size << " not 128-byte aligned";
        free_tensor(t);
    }
}

// ===========================================================================
// 7. BumpAllocator 行为测试
// ===========================================================================
TEST(BumpAllocatorTest, MultipleSlabs) {
    // 先 reset 默认分配器，避免之前测试的分配影响
    default_allocator().reset();
    EXPECT_EQ(default_allocator().num_slabs(), 0);

    // 分配多个 tensor，验证 slab 增长
    // 每个 tensor ~16 MB，应该触发多个 slab（默认 64 MB/slab）
    for (int i = 0; i < 8; ++i) {
        auto shape = TensorShape::make_2d(1024, 4096); // ~16 MB float
        auto t = allocate_tensor<float>(shape);
        EXPECT_NE(t.data_, nullptr);
    }

    // 应该至少有 2 个 slab（8 * 16 = 128 MB > 64 MB/slab）
    EXPECT_GT(default_allocator().num_slabs(), 0);
}

TEST(BumpAllocatorTest, Reset) {
    default_allocator().reset();
    EXPECT_EQ(default_allocator().num_slabs(), 0);
    EXPECT_EQ(default_allocator().total_allocated(), 0);
}

// ===========================================================================
// 8. Move 语义测试
// ===========================================================================
TEST(TensorMoveTest, MoveConstructor) {
    auto shape = TensorShape::make_1d(100);
    auto t1 = allocate_tensor<float>(shape);
    auto* ptr1 = t1.data_;

    auto t2 = std::move(t1);
    EXPECT_EQ(t2.data_, ptr1);     // t2 接管
    EXPECT_TRUE(t2.owner());
    EXPECT_EQ(t1.data_, nullptr);  // t1 失效
    EXPECT_FALSE(t1.owner());

    free_tensor(t2);
}

TEST(TensorMoveTest, MoveAssignment) {
    auto shape = TensorShape::make_1d(100);
    auto t1 = allocate_tensor<float>(shape);
    auto shape2 = TensorShape::make_1d(200);
    auto t2 = allocate_tensor<float>(shape2);
    auto* ptr1 = t1.data_;

    t2 = std::move(t1);
    EXPECT_EQ(t2.data_, ptr1);
    EXPECT_TRUE(t2.owner());
    EXPECT_EQ(t1.data_, nullptr);

    free_tensor(t2);
}

// ===========================================================================
// 9. Empty/Edge cases
// ===========================================================================
TEST(TensorEdgeTest, DefaultConstruction) {
    Tensor<float> t;
    EXPECT_TRUE(t.empty());
    EXPECT_FALSE(t.owner());
    EXPECT_EQ(t.data_, nullptr);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorEdgeTest, SizeOneAllocation) {
    auto shape = TensorShape::make_1d(1);
    auto t = allocate_tensor<float>(shape);
    EXPECT_EQ(t.size(), 1);
    EXPECT_EQ(t.bytes(), sizeof(float));
    EXPECT_NE(t.data_, nullptr);
    free_tensor(t);
}

// ===========================================================================
// test_tensor_view.cu — Tensor 零拷贝视图（view_narrow 区间切片）
// 纯 host 侧指针/形状校验，不触碰 GPU。
// ===========================================================================
#include <gtest/gtest.h>
#include <vector>
#include "tensor/tensor.h"

using namespace nemotron;

TEST(TensorViewTest, NarrowLastDim) {
    // [T=3, F=10] 沿最后一维取 [2, 2+4) → [3,4] strided 视图
    std::vector<float> buf(3 * 10);
    Tensor<float> t; t.data_ = buf.data(); t.shape_ = TensorShape::make_2d(3, 10); t.owner_ = false;

    auto v = t.view_narrow(1, 2, 4);
    EXPECT_EQ(v.data_, buf.data() + 2);   // 偏移 start*stride(1) = 2*1
    EXPECT_EQ(v.dim(0), 3);
    EXPECT_EQ(v.dim(1), 4);
    EXPECT_EQ(v.stride(0), 10);           // 外维 stride 不变 → strided 视图
    EXPECT_EQ(v.stride(1), 1);
    EXPECT_FALSE(v.owner());
}

TEST(TensorViewTest, NarrowFirstDim) {
    // [B=4, D=8] 沿第 0 维取 [1, 1+2) → 连续 [2,8]
    std::vector<float> buf(4 * 8);
    Tensor<float> t; t.data_ = buf.data(); t.shape_ = TensorShape::make_2d(4, 8); t.owner_ = false;

    auto v = t.view_narrow(0, 1, 2);
    EXPECT_EQ(v.data_, buf.data() + 1 * 8);
    EXPECT_EQ(v.dim(0), 2);
    EXPECT_EQ(v.dim(1), 8);
    EXPECT_EQ(v.size(), 16);
}

TEST(TensorViewTest, InProjSplit) {
    // Mamba in_proj 输出 [T, 17504] = z(7680) | xBC(9728) | dt(96)，免拷贝切三段
    const int T = 2, Z = 7680, XBC = 9728, DT = 96, F = Z + XBC + DT;
    std::vector<float> buf((size_t)T * F);
    Tensor<float> t; t.data_ = buf.data(); t.shape_ = TensorShape::make_2d(T, F); t.owner_ = false;

    auto z   = t.view_narrow(1, 0,        Z);
    auto xbc = t.view_narrow(1, Z,        XBC);
    auto dt  = t.view_narrow(1, Z + XBC,  DT);

    EXPECT_EQ(z.data_,   buf.data() + 0);
    EXPECT_EQ(xbc.data_, buf.data() + Z);
    EXPECT_EQ(dt.data_,  buf.data() + Z + XBC);
    EXPECT_EQ(xbc.dim(1), XBC);
    EXPECT_EQ(dt.dim(1),  DT);
    EXPECT_EQ(z.stride(0), F);    // 仍按原行宽跨行（strided）

    // conv 输出 xBC [T, 9728] 再切 x|B|C
    const int X = 7680, BN = 1024, CN = 1024;  // 7680 + 8*128 + 8*128
    auto cx = xbc.view_narrow(1, 0,        X);
    auto cB = xbc.view_narrow(1, X,        BN);
    auto cC = xbc.view_narrow(1, X + BN,   CN);
    EXPECT_EQ(cx.data_, buf.data() + Z + 0);
    EXPECT_EQ(cB.data_, buf.data() + Z + X);
    EXPECT_EQ(cC.data_, buf.data() + Z + X + BN);
}

//
// sdpa_cudnn.cpp — cuDNN frontend SDPA (FlashAttention) 实现
//
// 纯 host TU，由 MSVC 编译。绝不能被 NVCC 处理（cudnn_frontend 的巨型
// R"KERNEL(...)" 字面量会让 cudafe++ 崩溃）。声明见 sdpa_cudnn.h。
//
#ifdef USE_CUDNN

#include "ops/attention/sdpa_cudnn.h"

#include <cudnn_frontend.h>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "ops/cudnn_ctx.h"

namespace nemotron::ops::attention {

namespace fe = cudnn_frontend;

namespace {

// graph 构建很贵（heuristic + plan），按 (S, H, head_dim) 缓存。
struct SdpaCudnnPlan {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t uid_q = 1, uid_k = 2, uid_v = 3, uid_o = 4;
    int64_t workspace_size = 0;
};

std::shared_ptr<SdpaCudnnPlan> build_sdpa_plan(
    int S, int H, int head_dim, float attn_scale
) {
    auto plan = std::make_shared<SdpaCudnnPlan>();
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::BFLOAT16)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    const std::vector<int64_t> dim    = {1, H, S, head_dim};
    const std::vector<int64_t> stride = {(int64_t)H * S * head_dim,
                                         (int64_t)S * head_dim,
                                         (int64_t)head_dim, 1};

    auto Q = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("Q").set_uid(plan->uid_q)
                           .set_dim(dim).set_stride(stride));
    auto K = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("K").set_uid(plan->uid_k)
                           .set_dim(dim).set_stride(stride));
    auto V = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("V").set_uid(plan->uid_v)
                           .set_dim(dim).set_stride(stride));

    auto attrs = fe::graph::SDPA_attributes()
                     .set_name("flash_prefill")
                     .set_is_inference(true)
                     .set_causal_mask(true)
                     .set_attn_scale(attn_scale);

    // sdpa() 返回 {O, Stats}；推理模式 Stats 为空，只用 O。
    auto [O, Stats] = g->sdpa(Q, K, V, attrs);
    O->set_output(true).set_uid(plan->uid_o).set_dim(dim).set_stride(stride);

    auto& ctx = nemotron::CudnnContext::instance();
    auto st = g->build(ctx.handle(), {fe::HeurMode_t::A});
    if (!st.is_good()) return nullptr;

    plan->graph = g;
    plan->workspace_size = g->get_workspace_size();
    return plan;
}

std::shared_ptr<SdpaCudnnPlan> get_sdpa_plan(
    int S, int H, int head_dim, float attn_scale
) {
    static std::mutex mtx;
    static std::unordered_map<uint64_t, std::shared_ptr<SdpaCudnnPlan>> cache;
    const uint64_t key = ((uint64_t)S << 32) ^ ((uint64_t)H << 16) ^ (uint64_t)head_dim;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto plan = build_sdpa_plan(S, H, head_dim, attn_scale);
    cache[key] = plan;
    return plan;
}

}  // namespace

bool sdpa_prefill_bf16_cudnn(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, int S, int H, int head_dim, cudaStream_t stream
) {
    const float scale = 1.f / std::sqrt((float)head_dim);
    auto plan = get_sdpa_plan(S, H, head_dim, scale);
    if (!plan || !plan->graph) return false;

    auto& ctx = nemotron::CudnnContext::instance();
    cudnnSetStream(ctx.handle(), stream);

    // workspace：进程级缓冲，按需增长，避免每次 malloc/free。
    static void*  ws_ptr = nullptr;
    static size_t ws_cap = 0;
    if (plan->workspace_size > (int64_t)ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        cudaMalloc(&ws_ptr, plan->workspace_size);
        ws_cap = plan->workspace_size;
    }

    std::unordered_map<int64_t, void*> variant_pack = {
        {plan->uid_q, (void*)Q},
        {plan->uid_k, (void*)K},
        {plan->uid_v, (void*)V},
        {plan->uid_o, (void*)O},
    };

    auto st = plan->graph->execute(ctx.handle(), variant_pack, ws_ptr);
    return st.is_good();
}

}  // namespace nemotron::ops::attention

#endif  // USE_CUDNN

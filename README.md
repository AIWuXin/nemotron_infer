# nemotron_infer

> 从零手写 CUDA 的 **NVIDIA-Nemotron-3-Nano-4B-FP8** 推理引擎。
> 不依赖 PyTorch 算子、不依赖 transformers、不依赖 llama.cpp —— 每一个 kernel（GEMM/GEMV、
> Mamba-2 SSD scan、causal conv1d、FlashAttention、RMSNorm…）都自己写，单张 **8GB RTX 4060** 跑通。

| | |
|---|---|
| **模型** | NVIDIA-Nemotron-3-Nano-4B-FP8（Mamba-2 / Attention / MLP 混合架构，42 层） |
| **精度** | fp8 (e4m3) 权重常驻，激活 bf16，归约/状态 fp32 |
| **显存** | ~5.5 GB（权重 fp8 常驻 + KV/状态） |
| **decode** | **~46 token/s**（贪心与采样持平，撞带宽天花板：5.4GB / 272GB·s⁻¹ ≈ 50 t/s 上限，实测 92%） |
| **平台** | Windows 11 + CUDA 12.8 + MSVC 2022 + Ninja，sm_89 / sm_120 |

对标：用户实测同卡 LM Studio Q8-GGUF = 50 t/s。本引擎已贴着同一条带宽墙。

---

## 亮点

- **全手写 kernel**：FP8/BF16/FP32 三精度的 GEMM，自定义 **fp8 W8A16 GEMV**（decode M=1 快路），
  Mamba-2 **chunked SSD scan**（TF32 tensor core）+ 串行 fp32 参考路径，**手写 FlashAttention**
  （在线 softmax，NoPE GQA），causal conv1d、relu²、RMSNorm / gated-RMSNorm、embedding、lm_head。
- **逐算子对齐 HF 金标准**：每个 block 的 fp32/bf16/fp8 路径都与 HuggingFace 参考实现逐元素比对
  （`tools/dump_*.py` 产出 golden，C++ gtest 比对）；整模型 logits 与纯 torch 参考 argmax 一致、rel_l2 ≈ 3.8%。
- **混合架构正确落地**：21 个 Mamba-2 层（O(1) 状态，不随上下文增长）+ 4 个 **NoPE** Attention 层
  （GQA 40Q/8KV，KV cache）+ 17 个 relu² MLP 层，按 `hybrid_override_pattern` 接线。
- **Python 编排 + CUDA 计算**：权重加载、层循环、采样在 Python（极简、易对齐）；所有重活在手写 kernel。
- **交互式终端**：rich 渲染的思维链 + 回答流式输出，GPU 上采样（softmax/top-p/multinomial 全在显卡）。

---

## 架构速览

```text
backbone.embeddings → 42 × NemotronHBlock → norm_f → lm_head
                         │
   每层 pre-norm 残差： out = x + mixer(rmsnorm(x))
   mixer 三选一：
     • Mamba-2 : in_proj → causal_conv1d+SiLU → SSD scan → gated-RMSNorm → out_proj
     • Attention: q/k/v_proj → 因果 SDPA(GQA, NoPE 无 RoPE) → o_proj
     • MLP      : up_proj → relu²(ReLU(x)²) → down_proj
```

关键事实（容易踩坑）：

- **Attention 是 NoPE**：位置信息全部来自 Mamba 层，注意力层不加 RoPE。
- **GQA**：40 个 Q 头 / 8 个 KV 头（group=5），head_dim=128，scale=1/√128，无 bias。
- **MLP 无门控**：`down(relu(up(x))²)`，intermediate=12544。
- **FP8 权重格式**（modelopt）：`weight`(e4m3) + per-tensor 标量 `weight_scale`；引擎加载时广播成 per-row scale。
- **decode 状态 O(1)**：Mamba 层用 conv_state+ssm_state（定长），仅 4 个 Attention 层的 KV cache 随长度增长。

---

## 环境要求

- **OS**：Windows 11（构建脚本按 MSVC + Windows 路径写；Linux 需自行调整）
- **CUDA**：12.8（`nvcc` 在 PATH）
- **编译器**：Visual Studio 2022（MSVC）+ Ninja + CMake ≥ 3.20
- **GPU**：Compute Capability 8.9（RTX 4060/4090 等）或 12.0；默认 `CMAKE_CUDA_ARCHITECTURES=89;120`
- **Python**：3.12（用 [uv](https://docs.astral.sh/uv/) 管理）
- **显存**：≥ 8 GB

---

## 快速开始

### 1. 准备模型权重

把 `NVIDIA-Nemotron-3-Nano-4B-FP8` 的全部文件（`*.safetensors`、`tokenizer.json`、`config.json` 等）
放到：

```text
model/NVIDIA-Nemotron-3-Nano-4B-FP8/
```

### 2. 安装 Python 依赖（含 CUDA 版 torch）

```powershell
uv sync                 # 创建 .venv，装运行依赖 + pybind11
uv sync --group dev     # 额外装 torch(cu128)/safetensors/tokenizers/rich（引擎与校验需要）
```

> torch 必须是 CUDA 轮子（pyproject 已锁 `pytorch-cu128` 索引）——引擎用 `data_ptr()` 把显存指针
> 喂给手写 kernel，CPU-only 的 torch 不可用。后续命令统一加 `--no-sync` 防止 uv 回退成 +cpu 轮子。

### 3. 编译 CUDA 扩展（pybind 模块）

CMake 会在 `.venv` 里找 Python/pybind11，所以先完成第 2 步。Windows 下 PowerShell 不自动加载
MSVC 环境，需先注入 vcvars：

```powershell
# 注入 MSVC 工具链环境
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && set" `
  | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] } }

# 配置 + 构建 pybind 模块（产物 binding.*.pyd 自动拷到 nemotron_infer/csrc/）
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target nemotron_infer -j 18
```

### 4. 跑起来

```powershell
# 交互式终端（思维链 + 回答流式渲染）
uv run --no-sync python -m nemotron_infer.chat

# 引擎自测 + 实测 t/s
uv run --no-sync python -m nemotron_infer.engine
```

---

## Python API

```python
from nemotron_infer import NemotronEngine

eng = NemotronEngine(max_tokens=16384)      # max_tokens = prefill 上限 + KV cache 容量

# 一次性贪心生成
tokens, prefill_ms, dec_ms = eng.generate(prompt_ids, n_new=128)

# 流式采样（逐 token yield id；末尾 yield 含计时的 dict）
for tok in eng.stream_generate(prompt_ids, temperature=0.6, top_p=0.95,
                               rep_penalty=1.1, stop_ids=(11,)):
    if isinstance(tok, dict):
        print(tok)          # {'prefill_ms':..., 'decode_tps':..., 'n':...}
    else:
        ...                 # token id

eng.reset_states()          # 新会话前清 conv/ssm/KV
```

> ⚠️ **必须采样、不要贪心做对话**：reasoning 模型按采样训练，贪心会让小模型重复死循环或莫名拒答。
> 实测稳定档：`temperature=0.6 + top_p=0.95 + rep_penalty=1.1 + 近 64 token 窗口`。

终端命令：`/exit` 退出 · `/clear` 清空对话 · `/system <内容>` 设系统提示。

---

## 项目结构

```text
nemotron_infer/
  engine.py            整模型 Python 编排引擎（加载/prefill/decode/采样）
  chat.py              rich 交互式终端（ChatML 思维链渲染）
  __init__.py          导出 NemotronEngine
  csrc/
    include/
      binding.cu              pybind 入口：暴露各 block 的 forward/decode
      tensor/                 Tensor + BumpAllocator（软回退，零热路径 cudaMalloc）
      ops/                    手写算子
        gemm.cuh / gemv.cuh   FP32/BF16/FP8 GEMM + fp8 W8A16 GEMV
        mamba2/               causal_conv1d, ssd_scan（串行/chunked/fused）, ssm
        attention/            sdpa_prefill, sdpa_decode（手写 FlashAttention）, cudnn 备选
        reduce/elementwise/embedding.cuh   RMSNorm, relu², add, embedding, lm_head
      model/                  mamba_block / attention_block / mlp_block 三类层装配
tests/                 gtest：ops 算子级 + model 层级，对齐 HF golden
tools/                 dump_*（产 golden）, ref_full_model, validate_full, bench/prof
doc/                   精度表、算子规划、优化路线图
CMakeLists.txt         pybind 模块 + 三个测试可执行（TEST_TENSOR/TEST_OPS/TEST_MODEL）
```

---

## 测试与验证

```powershell
# 算子级 + 层级单测（先产 golden，再编译，再跑）
uv run --no-sync python tools/dump_mamba_block.py
uv run --no-sync python tools/dump_mlp_block.py
uv run --no-sync python tools/dump_attention_block.py
cmake --build cmake-build-release --target TEST_OPS TEST_MODEL -j 18
.\cmake-build-release\TEST_OPS.exe        # 算子全套
.\cmake-build-release\TEST_MODEL.exe      # 三类 block 对齐 HF

# 整模型 logits 对齐纯 torch 参考
uv run --no-sync python tools/ref_full_model.py     # 产参考 logits
uv run --no-sync python tools/validate_full.py      # 比对 argmax / top-k / rel_l2

# 性能
uv run --no-sync python tools/bench_decode.py       # decode t/s
uv run --no-sync python tools/prof_decode.py        # 逐层类型 wall vs GPU 时间
```

---

## 已知限制

- **单卡单流、batch=1**：面向本地单用户交互；无连续批处理 / paged-attention。
- **上下文上限**：模型原生支持 256K，但本引擎 KV cache 随 `max_tokens` 线性吃显存
  （4 个 attn 层 ≈ 16KB/token），8GB 卡实际上限约 **16–32K**。
- **仅 Windows 构建脚本**：核心 kernel 与 CMake 跨平台，但 vcvars / DLL 拷贝逻辑按 Windows 写。
- **prefill 长序列**：超长 prompt 的 chunked SSD scan 已上 tensor core，但仍受带宽限制。

---

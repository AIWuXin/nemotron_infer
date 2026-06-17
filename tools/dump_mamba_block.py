"""
dump_mamba_block.py — 生成单个 Mamba NemotronHBlock 的 HF 金标准参考 dump。

仅 dev 用（uv dev 依赖 torch）。release 推理不需要。

为避免引入 transformers / mamba_ssm（后者需 CUDA/triton 编译），这里**不 import HF
modeling 文件**，而是把 modeling_nemotron_h.py 里的 chunked SSD 数学（segment_sum /
reshape_into_chunks / 分块扫描，行 93-141 + 619-691）**逐行 verbatim 抄过来**，
配纯 torch 的 gated RMSNorm（= mamba_ssm rms_norm_ref, norm_before_gate=False）。
数学上与 HF 参考一致，但只依赖 torch+numpy、CPU 可跑。

输出：tests/data/mamba_block/ 下每个张量一个 raw little-endian float32 .bin + meta.txt。
C++ 测试（tests/ops/test_mamba_block.cu）用相同的小 config 常量加载比对。

跑法：uv sync  &&  uv run python tools/dump_mamba_block.py
"""
import os
import struct
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(1234)
DT = torch.float32

# ---- 小尺寸 config（保持 in_proj 公式使 d_mlp=0；S 是 chunk 的整数倍，免 padding 分支）----
HIDDEN = 64
H = 4            # mamba_num_heads
P = 16           # mamba_head_dim
N = 16           # ssm_state_size
G = 2            # n_groups
CONV_K = 4
CHUNK = 8
S = 24           # 3 个 chunk，验证 inter-chunk 递归
B = 1
EPS = 1e-5

INTER = H * P                       # 64
CONV_DIM = INTER + 2 * G * N        # 128
PROJ = INTER + CONV_DIM + H         # 196
GROUP_SIZE = INTER // G             # 32
DT_MIN, DT_MAX = 0.0, float("inf")  # 本模型 dt_limit=(0,inf)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "mamba_block")
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# verbatim from modeling_nemotron_h.py（行 93-141）
# ===========================================================================
def pad_tensor_by_size(input_tensor, pad_size):
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


# ===========================================================================
# 纯 torch gated RMSNorm = mamba_ssm rms_norm_ref(norm_before_gate=False, grouped)
#   gated = x * silu(z); rstd = rsqrt(mean_group(gated^2)+eps); out = gated*rstd*weight
# ===========================================================================
def gated_rmsnorm(x, weight, z, group_size, eps):
    x = x.float() * F.silu(z.float())
    xg = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    rstd = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    out = (xg * rstd).reshape(x.shape) * weight
    return out


def plain_rmsnorm(x, weight, eps):
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * weight


# ===========================================================================
# 权重（nn.Linear 权重布局 [out, in]）
# ===========================================================================
block_norm_w = torch.randn(HIDDEN, dtype=DT)
in_proj_w = torch.randn(PROJ, HIDDEN, dtype=DT) * 0.1
conv1d_w = torch.randn(CONV_DIM, 1, CONV_K, dtype=DT) * 0.2   # depthwise: [conv_dim,1,K]
conv1d_b = torch.randn(CONV_DIM, dtype=DT) * 0.1
A_log = torch.randn(H, dtype=DT) * 0.5 - 1.0
D_param = torch.rand(H, dtype=DT) * 0.1
dt_bias = torch.rand(H, dtype=DT) * 0.2
gnorm_w = torch.randn(INTER, dtype=DT)
out_proj_w = torch.randn(HIDDEN, INTER, dtype=DT) * 0.1

x_input = torch.randn(B, S, HIDDEN, dtype=DT)


# ===========================================================================
# 前向（NemotronHBlock: residual + mixer(prenorm(x))；mixer = torch_forward 慢路径）
# ===========================================================================
DBG = {}

def mamba_block_forward(input_states):
    residual = input_states
    hs = plain_rmsnorm(input_states, block_norm_w, EPS)            # pre-norm
    DBG["normed"] = hs

    # in_proj
    proj = F.linear(hs, in_proj_w)                                 # [B,S,PROJ]
    gate, hidden_states_B_C, dt = torch.split(proj, [INTER, CONV_DIM, H], dim=-1)
    DBG["gate"] = gate; DBG["xbc"] = hidden_states_B_C; DBG["dt_raw"] = dt

    # depthwise causal conv1d + silu（verbatim 行 549 语义）
    hbc = hidden_states_B_C.transpose(1, 2)                        # [B,CONV_DIM,S]
    hbc = F.conv1d(hbc, conv1d_w, bias=conv1d_b, padding=CONV_K - 1, groups=CONV_DIM)[..., :S]
    hbc = F.silu(hbc.transpose(1, 2))                              # [B,S,CONV_DIM]
    DBG["xbc_conv"] = hbc

    x, Bm, Cm = torch.split(hbc, [INTER, G * N, G * N], dim=-1)

    # ---- chunked SSD（verbatim modeling 行 619-691，no cache）----
    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, DT_MIN, DT_MAX)
    hidden_states = x.reshape(B, S, -1, P).float()                 # [B,S,H,P]
    Bm = Bm.reshape(B, S, -1, N).float()                          # [B,S,G,N]
    Cm = Cm.reshape(B, S, -1, N).float()
    # ⚠️ HF naive 写的是 .repeat(tile, head h→group h%G)，但这与 Mamba2=GQA 的分组语义
    # （连续 head 共享一组，= repeat_interleave，head h→group h//(H/G)）不一致；训练用的
    # fast path mamba_chunk_scan_combined 用的是 interleave。故这里改 repeat_interleave 以
    # 对齐真实模型与本引擎的 GQA 约定（kv_head=q_head/(Hq/Hkv)）。
    Bm = Bm.repeat_interleave(H // G, dim=2)                       # [B,S,H,N]
    Cm = Cm.repeat_interleave(H // G, dim=2)
    pad_size = (CHUNK - S % CHUNK) % CHUNK

    A = -torch.exp(A_log.float())                                  # [H]
    D_residual = D_param[..., None] * pad_tensor_by_size(hidden_states, pad_size)
    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt                            # [B,S,H]

    hidden_states, A, Bm, Cm = [reshape_into_chunks(t, pad_size, CHUNK) for t in (hidden_states, A, Bm, Cm)]
    A = A.permute(0, 3, 1, 2)                                      # [B,H,nc,chunk]
    A_cumsum = torch.cumsum(A, dim=-1)

    Lm = torch.exp(segment_sum(A))
    G_int = Cm[:, :, :, None, :, :] * Bm[:, :, None, :, :, :]
    G_ = G_int.sum(dim=-1)
    M_int = G_[..., None] * Lm.permute(0, 2, 3, 4, 1)[..., None]
    M = M_int.sum(dim=-1)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = Bm * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states = new_states[:, :-1]

    state_decay_out = torch.exp(A_cumsum)
    C_times_states = Cm[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    y = Y_diag + Y_off
    y = y.reshape(B, -1, H, P)
    y = y + D_residual
    if pad_size > 0:
        y = y[:, :S, :, :]
    y = y.reshape(B, S, -1)                                        # [B,S,INTER]
    DBG["scan_y"] = y

    # gated RMSNorm
    y = gated_rmsnorm(y, gnorm_w, gate, GROUP_SIZE, EPS)
    DBG["gnormed"] = y

    # out_proj + residual
    mixer_out = F.linear(y, out_proj_w)                           # [B,S,HIDDEN]
    DBG["mixer_out"] = mixer_out
    return residual + mixer_out


with torch.no_grad():
    expected = mamba_block_forward(x_input)


# ===========================================================================
# dump（raw little-endian float32）
# ===========================================================================
def save(name, t):
    arr = t.detach().contiguous().float().cpu().numpy().astype("<f4").ravel()
    arr.tofile(os.path.join(OUT_DIR, name + ".bin"))
    return arr.size


sizes = {}
sizes["input"]        = save("input", x_input)                       # [B*S, HIDDEN]
sizes["block_norm_w"] = save("block_norm_w", block_norm_w)
sizes["in_proj_w"]    = save("in_proj_w", in_proj_w)                 # [PROJ, HIDDEN]
sizes["conv1d_w"]     = save("conv1d_w", conv1d_w.reshape(CONV_DIM, CONV_K))  # [CONV_DIM, K]
sizes["conv1d_b"]     = save("conv1d_b", conv1d_b)
sizes["A_log"]        = save("A_log", A_log)
sizes["D"]            = save("D", D_param)
sizes["dt_bias"]      = save("dt_bias", dt_bias)
sizes["gnorm_w"]      = save("gnorm_w", gnorm_w)
sizes["out_proj_w"]   = save("out_proj_w", out_proj_w)              # [HIDDEN, INTER]
sizes["expected"]     = save("expected", expected)                  # [B*S, HIDDEN]
# 中间张量（debug 分阶段定位）
for k in ["normed", "gate", "xbc", "dt_raw", "xbc_conv", "scan_y", "gnormed", "mixer_out"]:
    sizes[k] = save(k, DBG[k])

with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
    f.write(f"HIDDEN={HIDDEN} H={H} P={P} N={N} G={G} CONV_K={CONV_K} CHUNK={CHUNK} "
            f"S={S} B={B} INTER={INTER} CONV_DIM={CONV_DIM} PROJ={PROJ} "
            f"GROUP_SIZE={GROUP_SIZE} EPS={EPS}\n")
    for k, v in sizes.items():
        f.write(f"{k}={v}\n")

print("dumped to", os.path.abspath(OUT_DIR))
print("expected[0,:5] =", expected.reshape(-1)[:5].tolist())
for k, v in sizes.items():
    print(f"  {k}: {v} floats")

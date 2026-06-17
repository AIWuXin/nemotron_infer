"""
dump_attention_block.py — 生成单个 Attention NemotronHBlock 的 HF 金标准参考 dump。

⚠️ Nemotron-H attention 是 NoPE（无 RoPE）：见 modeling_nemotron_h.py NemotronHAttention.forward，
   全程无 rotary。纯 q/k/v_proj → causal SDPA(GQA, scale=1/sqrt(HEAD)) → o_proj。
外层 block: out = input + mixer(rmsnorm(input))。

⚠️ HEAD 必须=128：本引擎 SDPA prefill/decode 核硬编码 head_dim=128
   （prefill: out[4]×32lane；decode: DPL=HEAD/32）。故小 config 只缩小 HIDDEN/H/S，
   保持 HEAD=128。

输出 tests/data/attention_block/ 下 raw little-endian float32 .bin + meta.txt。
跑法：uv run python tools/dump_attention_block.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(777)
DT = torch.float32

HIDDEN = 256
H_Q = 4
H_KV = 2          # GQA group = H_Q/H_KV = 2
HEAD = 128        # 必须 128（SDPA 核约束）
S = 40            # >32：强制 SDPA prefill 走多 KV tile（覆盖跨 tile online-softmax 路径）
B = 1
EPS = 1e-5

QD = H_Q * HEAD   # 512
KD = H_KV * HEAD  # 256

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "attention_block")
os.makedirs(OUT_DIR, exist_ok=True)


def plain_rmsnorm(x, weight, eps):
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * weight


def repeat_kv(hidden_states, n_rep):
    b, h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return hidden_states.reshape(b, h * n_rep, s, d)


# 权重（nn.Linear 布局 [out, in]），无 bias
block_norm_w = torch.randn(HIDDEN, dtype=DT)
q_proj_w = torch.randn(QD, HIDDEN, dtype=DT) * 0.1
k_proj_w = torch.randn(KD, HIDDEN, dtype=DT) * 0.1
v_proj_w = torch.randn(KD, HIDDEN, dtype=DT) * 0.1
o_proj_w = torch.randn(HIDDEN, QD, dtype=DT) * 0.1
x_input = torch.randn(B, S, HIDDEN, dtype=DT)

DBG = {}


def attention_block_forward(input_states):
    residual = input_states
    hs = plain_rmsnorm(input_states, block_norm_w, EPS)
    DBG["normed"] = hs

    q = F.linear(hs, q_proj_w).view(B, S, H_Q, HEAD).transpose(1, 2)   # [B,H_Q,S,HEAD]
    k = F.linear(hs, k_proj_w).view(B, S, H_KV, HEAD).transpose(1, 2)  # [B,H_KV,S,HEAD]
    v = F.linear(hs, v_proj_w).view(B, S, H_KV, HEAD).transpose(1, 2)
    k = repeat_kv(k, H_Q // H_KV)
    v = repeat_kv(v, H_Q // H_KV)

    # NoPE：直接 causal SDPA，scale 默认 1/sqrt(HEAD)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)     # [B,H_Q,S,HEAD]
    attn = attn.transpose(1, 2).contiguous().view(B, S, QD)
    DBG["attn"] = attn

    o = F.linear(attn, o_proj_w)                                       # [B,S,HIDDEN]
    DBG["o"] = o
    return residual + o


with torch.no_grad():
    expected = attention_block_forward(x_input)


def save(name, t):
    arr = t.detach().contiguous().float().cpu().numpy().astype("<f4").ravel()
    arr.tofile(os.path.join(OUT_DIR, name + ".bin"))
    return arr.size


sizes = {}
sizes["input"]        = save("input", x_input)
sizes["block_norm_w"] = save("block_norm_w", block_norm_w)
sizes["q_proj_w"]     = save("q_proj_w", q_proj_w)
sizes["k_proj_w"]     = save("k_proj_w", k_proj_w)
sizes["v_proj_w"]     = save("v_proj_w", v_proj_w)
sizes["o_proj_w"]     = save("o_proj_w", o_proj_w)
sizes["expected"]     = save("expected", expected)
for k in ["normed", "attn", "o"]:
    sizes[k] = save(k, DBG[k])

with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
    f.write(f"HIDDEN={HIDDEN} H_Q={H_Q} H_KV={H_KV} HEAD={HEAD} S={S} B={B} "
            f"QD={QD} KD={KD} EPS={EPS}\n")
    for k, v in sizes.items():
        f.write(f"{k}={v}\n")

print("dumped to", os.path.abspath(OUT_DIR))
print("expected[:5] =", expected.reshape(-1)[:5].tolist())

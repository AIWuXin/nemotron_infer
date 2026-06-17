"""
dump_mlp_block.py — 生成单个 MLP NemotronHBlock 的 HF 金标准参考 dump。

NemotronHMLP.forward: down_proj(relu²(up_proj(x)))，无门控；relu2 = ReLU(x)²。
外层 block: out = input + mixer(rmsnorm(input))。

输出 tests/data/mlp_block/ 下 raw little-endian float32 .bin + meta.txt。
跑法：uv run python tools/dump_mlp_block.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(2024)
DT = torch.float32

HIDDEN = 64
INTER = 256
S = 8
B = 1
EPS = 1e-5

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "mlp_block")
os.makedirs(OUT_DIR, exist_ok=True)


def plain_rmsnorm(x, weight, eps):
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * weight


# 权重（nn.Linear 布局 [out, in]）
block_norm_w = torch.randn(HIDDEN, dtype=DT)
up_proj_w = torch.randn(INTER, HIDDEN, dtype=DT) * 0.1
down_proj_w = torch.randn(HIDDEN, INTER, dtype=DT) * 0.1
x_input = torch.randn(B, S, HIDDEN, dtype=DT)

DBG = {}


def mlp_block_forward(input_states):
    residual = input_states
    hs = plain_rmsnorm(input_states, block_norm_w, EPS)
    DBG["normed"] = hs
    up = F.linear(hs, up_proj_w)
    DBG["up"] = up
    act = F.relu(up).pow(2)        # relu2 = ReLU(x)^2
    DBG["act"] = act
    down = F.linear(act, down_proj_w)
    DBG["down"] = down
    return residual + down


with torch.no_grad():
    expected = mlp_block_forward(x_input)


def save(name, t):
    arr = t.detach().contiguous().float().cpu().numpy().astype("<f4").ravel()
    arr.tofile(os.path.join(OUT_DIR, name + ".bin"))
    return arr.size


sizes = {}
sizes["input"]        = save("input", x_input)
sizes["block_norm_w"] = save("block_norm_w", block_norm_w)
sizes["up_proj_w"]    = save("up_proj_w", up_proj_w)
sizes["down_proj_w"]  = save("down_proj_w", down_proj_w)
sizes["expected"]     = save("expected", expected)
for k in ["normed", "up", "act", "down"]:
    sizes[k] = save(k, DBG[k])

with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
    f.write(f"HIDDEN={HIDDEN} INTER={INTER} S={S} B={B} EPS={EPS}\n")
    for k, v in sizes.items():
        f.write(f"{k}={v}\n")

print("dumped to", os.path.abspath(OUT_DIR))
print("expected[:5] =", expected.reshape(-1)[:5].tolist())

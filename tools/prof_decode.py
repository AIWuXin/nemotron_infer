"""prof_decode.py — 逐层类型测 decode 单步延迟，定位 112ms/tok 的去向。
跑法：uv run --no-sync python tools/prof_decode.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nemotron_infer.engine import NemotronEngine, ptr
import nemotron_infer.engine as E
B = E.B

eng = NemotronEngine()

# 先 prefill 建好状态（cap_states）
prompt = list(range(1, 40))
eng.reset_states()
eng.prefill(prompt, cap_states=True)
B.sync()

cur = eng.dbuf[0]; other = eng.dbuf[1]

# 找各类型第一层
mamba_ly = next(l for l in eng.layers if l['type'] == 'mamba')
attn_ly  = next(l for l in eng.layers if l['type'] == 'attn')
mlp_ly   = next(l for l in eng.layers if l['type'] == 'mlp')

n_mamba = sum(1 for l in eng.layers if l['type'] == 'mamba')
n_attn  = sum(1 for l in eng.layers if l['type'] == 'attn')
n_mlp   = sum(1 for l in eng.layers if l['type'] == 'mlp')

import torch
def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        B.reset_allocator(); fn()
    B.sync()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    t0 = time.time()
    ev0.record()
    for _ in range(iters):
        B.reset_allocator(); fn()
    ev1.record()
    B.sync()
    wall = (time.time() - t0) * 1000 / iters
    gpu = ev0.elapsed_time(ev1) / iters
    return wall, gpu  # ms/call (wall, gpu)

# 单 kernel 隔离测时
ids1 = torch.tensor([5], dtype=torch.int64, device=E.DEV)
w_emb, g_emb = bench(lambda: B.embedding(ptr(eng.embed), ptr(ids1), ptr(cur), 1))
w_rms, g_rms = bench(lambda: B.rmsnorm(ptr(cur), ptr(other), ptr(eng.norm_f), 1))
print(f"  embedding M=1: wall {w_emb:.3f} gpu {g_emb:.3f}")
print(f"  rmsnorm   M=1: wall {w_rms:.3f} gpu {g_rms:.3f}")

w_mamba, g_mamba = bench(lambda: eng._dec_mamba(mamba_ly, cur, other))
w_attn,  g_attn  = bench(lambda: eng._dec_attn(attn_ly, cur, other, 40))
w_mlp,   g_mlp   = bench(lambda: eng._run_mlp(mlp_ly, cur, other, 1))

def lmh():
    B.rmsnorm(ptr(cur), ptr(other), ptr(eng.norm_f), 1)
    B.lm_head(other.data_ptr(), ptr(eng.lm_head), ptr(eng.logits), 1)
w_lmh, g_lmh = bench(lmh)

print(f"  {'layer':12s} {'wall':>8s} {'gpu':>8s}  x cnt =  wall_sum  gpu_sum")
def row(name, w, g, cnt):
    print(f"  {name:12s} {w:7.3f}  {g:7.3f}  x {cnt:2d} = {w*cnt:8.2f} {g*cnt:8.2f}")
row('mamba', w_mamba, g_mamba, n_mamba)
row('attn',  w_attn,  g_attn,  n_attn)
row('mlp',   w_mlp,   g_mlp,   n_mlp)
row('lm_head', w_lmh, g_lmh, 1)
wsum = w_mamba*n_mamba + w_attn*n_attn + w_mlp*n_mlp + w_lmh
gsum = g_mamba*n_mamba + g_attn*n_attn + g_mlp*n_mlp + g_lmh
print(f"  ----------------------------------------")
print(f"  wall sum: {wsum:7.2f} ms → {1000/wsum:5.1f} t/s | gpu sum: {gsum:7.2f} ms → {1000/gsum:5.1f} t/s")

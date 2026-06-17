"""bench_decode.py — 实测 decode 单 token 延迟与 t/s（gemv 快路验证用）。
跑法：uv run --no-sync python tools/bench_decode.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nemotron_infer.engine import NemotronEngine

eng = NemotronEngine()

# 任意 prompt（greedy，纯测速）
prompt = [10, 9015, 1374, 525, 498, 30, 11]   # 随便几个 token
N_NEW = 128

toks, prefill_ms, dec_ms = eng.generate(prompt, N_NEW, greedy=True)
tps = 1000.0 / dec_ms if dec_ms > 0 else 0.0
print(f"prefill {len(prompt)} tok / {prefill_ms:.1f} ms")
print(f"decode  {N_NEW} tok | {dec_ms:.3f} ms/tok | {tps:.1f} tok/s")

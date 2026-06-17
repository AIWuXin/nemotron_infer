"""
validate_full.py — 引擎 fp8 prefill logits 对齐纯 torch fp32 参考。

先跑：uv run python tools/ref_full_model.py   （产出 tests/data/full_model/）
再跑：uv run python tools/validate_full.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nemotron_infer.engine import NemotronEngine

DATA = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'full_model')

ids = np.fromfile(os.path.join(DATA, 'input_ids.bin'), dtype='<i8').tolist()
ref = np.fromfile(os.path.join(DATA, 'logits_ref.bin'), dtype='<f4')

t0 = time.time()
eng = NemotronEngine()
print(f'引擎加载 {time.time()-t0:.1f}s, 显存 {__import__("torch").cuda.memory_allocated()/1e9:.2f} GB')

t0 = time.time()
got = eng.prefill(ids).cpu().numpy()  # prefill 现返回 GPU 张量，比对前搬 CPU
dt = time.time() - t0
print(f'prefill {len(ids)} tok in {dt*1000:.1f}ms')

# top-k 对比
import numpy as np
def topk(a, k=10): return set(np.argsort(a)[-k:][::-1].tolist())
r_top5 = np.argsort(ref)[-5:][::-1]
g_top5 = np.argsort(got)[-5:][::-1]
print('参考 top5:', r_top5.tolist())
print('引擎 top5:', g_top5.tolist())
print('argmax  参考=%d  引擎=%d  %s' % (int(ref.argmax()), int(got.argmax()),
      'MATCH' if ref.argmax()==got.argmax() else 'MISMATCH'))
for k in (1, 5, 10):
    inter = len(topk(ref, k) & topk(got, k))
    print(f'  top{k} 重合 {inter}/{k}')

# 数值
sse = float(((got - ref)**2).sum()); ssr = float((ref**2).sum())
print('rel_l2 = %.4f' % (np.sqrt(sse/max(ssr,1e-12))))
# Spearman-ish：参考 top20 在引擎里的排名
order = np.argsort(got)[::-1]
rank = {t: i for i, t in enumerate(order.tolist())}
r20 = np.argsort(ref)[-20:][::-1]
print('参考 top20 token 在引擎中的排名:', [rank[int(t)] for t in r20])

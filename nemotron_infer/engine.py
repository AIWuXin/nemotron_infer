"""
engine.py — Nemotron-H 整模型 Python 编排引擎（fp8 常驻 + bf16 exclude 层）。

整模型推理的"指挥层"：权重常驻 GPU，Python 持有 hidden ping-pong / KV / conv / ssm
状态，逐层调用手写 CUDA block（`nemotron_infer.csrc.binding` pybind 模块），不依赖
PyTorch 的算子，只借 torch 管理显存指针与采样。

精度派发（按权重存储 dtype，加载期一次判定）：
  - fp8 矩阵权重     → e4m3 + per-row scale，走 fp8 block（in/out_proj、q/k/v/o、up/down）
  - exclude 层 / 嵌入 → bf16，走 bf16 block
  - norm / conv / A_log / D / dt_bias → fp32
层类型（mamba / attn / mlp）由权重命名存在性判定（in_proj→mamba，q_proj→attn，up_proj→mlp）。

显存编排约定：
  - 每层 forward/decode 前调 `B.reset_allocator()`（软回退，零 cudaMalloc/cudaFree），
    block 瞬态 workspace 走全局 BumpAllocator，输出写到调用方持久 buffer（Python ping-pong）。
  - decode 单 token 状态 O(1)：mamba 用 conv_state+ssm_state，attn 用增长的 KV cache。
  - 采样在 GPU 上做（见 `_sample`/`_lm_head_last`），每 token 只回传 1 个 token id。

公共 API：`prefill` / `decode_step` / `generate`（贪心）/ `stream_generate`（流式采样）。

跑法：uv run --no-sync python -m nemotron_infer.engine   # 自测 + t/s
"""
import os, sys, glob, time
import torch
from safetensors import safe_open

_HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin')
sys.path.insert(0, os.path.join(_HERE, 'csrc'))
import nemotron_infer.csrc.binding as B

MODEL_DIR = os.path.join(_HERE, '..', 'model', 'NVIDIA-Nemotron-3-Nano-4B-FP8')
DEV = 'cuda'
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
F32 = torch.float32


def ptr(t):
    """torch 张量 → 设备指针（uintptr），喂给 pybind block 的 data_ptr 参数。"""
    return t.data_ptr()


class NemotronEngine:
    """Nemotron-3-Nano-4B-FP8 单流推理引擎（权重 fp8 常驻，~5.5GB 显存，8GB 卡可跑）。

    用法::

        eng = NemotronEngine(max_tokens=16384)
        # 一次性贪心生成
        tokens, prefill_ms, dec_ms = eng.generate(prompt_ids, n_new=128)
        # 流式采样（逐 token yield id，末尾 yield 计时 dict）
        for tok in eng.stream_generate(prompt_ids, temperature=0.6, top_p=0.95):
            ...

    多轮对话每轮调一次 prefill→decode；新会话先 `reset_states()` 清 conv/ssm/KV。
    """

    def __init__(self, model_dir=MODEL_DIR, max_tokens=2048):
        """加载权重常驻 GPU + 预分配持久 buffer。

        max_tokens：prefill 序列上限 + attn KV cache 容量。注意 KV cache 随该值线性
        吃显存（4 个 attn 层 ≈ 16KB/token），256K 在 8GB 上放不下，实际上限 ~16-32K。
        """
        self.max_tokens = max_tokens
        self.H = B.HIDDEN
        self._load(model_dir)
        # 持久 buffer
        self.hbuf = [torch.zeros(max_tokens, self.H, dtype=BF16, device=DEV) for _ in range(2)]
        self.dbuf = [torch.zeros(1, self.H, dtype=BF16, device=DEV) for _ in range(2)]
        self.logits = torch.zeros(1, B.VOCAB, dtype=BF16, device=DEV)
        self._alloc_states()

    def _alloc_states(self):
        """每层续接状态：mamba conv/ssm(fp32)，attn KV cache(bf16, [H_KV,max,HEAD])。"""
        for ly in self.layers:
            if ly['type'] == 'mamba':
                ly['conv_state'] = torch.zeros(B.CONV_DIM * (B.CONV_K - 1), dtype=F32, device=DEV)
                ly['ssm_state'] = torch.zeros(B.H * B.P * B.N, dtype=F32, device=DEV)
            elif ly['type'] == 'attn':
                ly['kcache'] = torch.zeros(B.H_KV * self.max_tokens * B.HEAD, dtype=BF16, device=DEV)
                ly['vcache'] = torch.zeros(B.H_KV * self.max_tokens * B.HEAD, dtype=BF16, device=DEV)

    def reset_states(self):
        """清零所有层的续接状态（conv/ssm/KV）。新会话或 /clear 前必调，否则带入上轮上下文。"""
        for ly in self.layers:
            if ly['type'] == 'mamba':
                ly['conv_state'].zero_(); ly['ssm_state'].zero_()
            elif ly['type'] == 'attn':
                ly['kcache'].zero_(); ly['vcache'].zero_()

    # -------------------------------------------------------------------
    def _load(self, md):
        """从 *.safetensors 加载权重上 GPU：fp8 矩阵保 e4m3+per-row scale，其余按用途转
        bf16/fp32。扫 backbone.layers.{i} 数层数，按权重名判层类型。加载后释放 CPU 副本。"""
        # 收集所有张量到 CPU，再按需转精度上 GPU
        raw = {}
        for fn in glob.glob(os.path.join(md, '*.safetensors')):
            with safe_open(fn, 'pt') as f:
                for k in f.keys():
                    raw[k] = f.get_tensor(k)
        self.raw = raw

        # 层数：扫 backbone.layers.{i}
        nlayers = 0
        while f'backbone.layers.{nlayers}.norm.weight' in raw:
            nlayers += 1
        self.nlayers = nlayers

        def g32(k):  # → fp32 GPU
            return raw[k].to(F32).contiguous().to(DEV)
        def gbf(k):  # → bf16 GPU
            return raw[k].to(BF16).contiguous().to(DEV)
        def gw(k):   # 矩阵权重：fp8 保持 e4m3+scale，bf16 直传
            w = raw[k + '.weight']
            if w.dtype == FP8:
                n = w.shape[0]
                scale = raw[k + '.weight_scale'].float().reshape(())  # 标量
                wscale = torch.full((n,), float(scale), dtype=F32, device=DEV)
                return ('fp8', w.contiguous().to(DEV), wscale)
            else:
                return ('bf16', w.to(BF16).contiguous().to(DEV), None)

        self.embed = gbf('backbone.embeddings.weight')
        self.norm_f = g32('backbone.norm_f.weight')
        self.lm_head = gbf('lm_head.weight')

        self.layers = []
        keep = []  # 防止 GPU 张量被 GC
        for i in range(nlayers):
            p = f'backbone.layers.{i}.'
            ly = {'norm': g32(p + 'norm.weight')}
            if p + 'mixer.in_proj.weight' in raw:
                ly['type'] = 'mamba'
                ly['in_proj'] = gw(p + 'mixer.in_proj')
                ly['out_proj'] = gw(p + 'mixer.out_proj')
                ly['conv_w'] = g32(p + 'mixer.conv1d.weight').reshape(B.CONV_DIM, B.CONV_K)
                ly['conv_b'] = g32(p + 'mixer.conv1d.bias')
                ly['A_log'] = g32(p + 'mixer.A_log')
                ly['D'] = g32(p + 'mixer.D')
                ly['dt_bias'] = g32(p + 'mixer.dt_bias')
                ly['gnorm'] = g32(p + 'mixer.norm.weight')
            elif p + 'mixer.q_proj.weight' in raw:
                ly['type'] = 'attn'
                ly['q'] = gw(p + 'mixer.q_proj')
                ly['k'] = gw(p + 'mixer.k_proj')
                ly['v'] = gw(p + 'mixer.v_proj')
                ly['o'] = gw(p + 'mixer.o_proj')
            elif p + 'mixer.up_proj.weight' in raw:
                ly['type'] = 'mlp'
                ly['up'] = gw(p + 'mixer.up_proj')
                ly['down'] = gw(p + 'mixer.down_proj')
            else:
                raise RuntimeError(f'layer {i} 未知类型')
            self.layers.append(ly)
        # 释放 CPU raw（已上 GPU 的留着）
        del self.raw
        torch.cuda.synchronize()

    # -------------------------------------------------------------------
    # ---- prefill runner（cap_states=True 时写续接状态）----
    def _run_mamba(self, ly, cur, out, S, cap_states=False):
        """prefill 一个 mamba 层：cur[S,H]→out[S,H]。cap_states=True 写末态供 decode 续接。"""
        ip, op = ly['in_proj'], ly['out_proj']
        cs = ptr(ly['conv_state']) if cap_states else 0
        ss = ptr(ly['ssm_state']) if cap_states else 0
        if ip[0] == 'fp8':
            B.mamba_forward_fp8(
                ptr(cur), ptr(ly['norm']), ptr(ip[1]), ptr(ip[2]),
                ptr(ly['conv_w']), ptr(ly['conv_b']), ptr(ly['A_log']), ptr(ly['D']),
                ptr(ly['dt_bias']), ptr(ly['gnorm']), ptr(op[1]), ptr(op[2]),
                ptr(out), 1, S, cs, ss)
        else:
            B.mamba_forward(
                ptr(cur), ptr(ly['norm']), ptr(ip[1]), ptr(ly['conv_w']), ptr(ly['conv_b']),
                ptr(ly['A_log']), ptr(ly['D']), ptr(ly['dt_bias']), ptr(ly['gnorm']),
                ptr(op[1]), ptr(out), 1, S, cs, ss)

    def _run_attn(self, ly, cur, out, S, cap_states=False):
        """prefill 一个 attention 层（NoPE GQA）。cap_states=True 把 KV 写进 cache 供 decode。"""
        q, k, v, o = ly['q'], ly['k'], ly['v'], ly['o']
        kc = ptr(ly['kcache']) if cap_states else 0
        vc = ptr(ly['vcache']) if cap_states else 0
        cap = self.max_tokens if cap_states else 0
        if q[0] == 'fp8':
            B.attn_forward_fp8(
                ptr(cur), ptr(ly['norm']),
                ptr(q[1]), ptr(q[2]), ptr(k[1]), ptr(k[2]),
                ptr(v[1]), ptr(v[2]), ptr(o[1]), ptr(o[2]),
                ptr(out), S, kc, vc, cap)
        else:
            B.attn_forward(
                ptr(cur), ptr(ly['norm']),
                ptr(q[1]), ptr(k[1]), ptr(v[1]), ptr(o[1]),
                ptr(out), S, kc, vc, cap)

    def _run_mlp(self, ly, cur, out, S):
        """MLP 层 up→relu²→down（无状态，prefill/decode 同一函数，decode 传 S=1 走 gemv 快路）。"""
        up, dn = ly['up'], ly['down']
        if up[0] == 'fp8':
            B.mlp_forward_fp8(
                ptr(cur), ptr(ly['norm']), ptr(up[1]), ptr(up[2]),
                ptr(dn[1]), ptr(dn[2]), ptr(out), S)
        else:
            B.mlp_forward(
                ptr(cur), ptr(ly['norm']), ptr(up[1]), ptr(dn[1]), ptr(out), S)

    # ---- decode runner（M=1，就地更新状态）----
    def _dec_mamba(self, ly, cur, out):
        """decode 一个 mamba 层（M=1）：就地更新 conv_state/ssm_state，O(1) 不随长度增长。"""
        ip, op = ly['in_proj'], ly['out_proj']
        if ip[0] == 'fp8':
            B.mamba_decode_fp8(
                ptr(cur), ptr(ly['norm']), ptr(ip[1]), ptr(ip[2]),
                ptr(ly['conv_w']), ptr(ly['conv_b']), ptr(ly['A_log']), ptr(ly['D']),
                ptr(ly['dt_bias']), ptr(ly['gnorm']), ptr(op[1]), ptr(op[2]),
                ptr(ly['conv_state']), ptr(ly['ssm_state']), ptr(out), 1)
        else:
            B.mamba_decode(
                ptr(cur), ptr(ly['norm']), ptr(ip[1]), ptr(ly['conv_w']), ptr(ly['conv_b']),
                ptr(ly['A_log']), ptr(ly['D']), ptr(ly['dt_bias']), ptr(ly['gnorm']),
                ptr(op[1]), ptr(ly['conv_state']), ptr(ly['ssm_state']), ptr(out), 1)

    def _dec_attn(self, ly, cur, out, s_cache):
        """decode 一个 attention 层（M=1）：新 token 的 KV 追加到 cache[s_cache]，再做因果注意力。"""
        q, k, v, o = ly['q'], ly['k'], ly['v'], ly['o']
        if q[0] == 'fp8':
            B.attn_decode_fp8(
                ptr(cur), ptr(ly['norm']),
                ptr(q[1]), ptr(q[2]), ptr(k[1]), ptr(k[2]),
                ptr(v[1]), ptr(v[2]), ptr(o[1]), ptr(o[2]),
                ptr(ly['kcache']), ptr(ly['vcache']), s_cache, self.max_tokens, ptr(out))
        else:
            B.attn_decode(
                ptr(cur), ptr(ly['norm']),
                ptr(q[1]), ptr(k[1]), ptr(v[1]), ptr(o[1]),
                ptr(ly['kcache']), ptr(ly['vcache']), s_cache, self.max_tokens, ptr(out))

    # -------------------------------------------------------------------
    def _lm_head_last(self, hbuf, S):
        """对 hbuf 的最后一行做 final norm + lm_head，返回 logits[VOCAB] (fp32 GPU)。
        ⚠️ 留在 GPU：采样(softmax/sort/multinomial)在显卡上跑，只回传选中的 1 个 token id，
           避免每 token 把 131072 词表搬到 CPU 再排序（实测 CPU nucleus 采样 ~18.8ms/tok，
           几乎和整层 GPU 前向一样贵）。需 CPU 时调用方自行 .cpu()。"""
        B.reset_allocator()
        tmp = self.hbuf[1] if hbuf is self.hbuf[0] else self.hbuf[0]
        B.rmsnorm(ptr(hbuf), ptr(tmp), ptr(self.norm_f), S)
        last = tmp.data_ptr() + (S - 1) * self.H * 2  # bf16=2字节
        B.lm_head(last, ptr(self.lm_head), ptr(self.logits), 1)
        B.sync()
        return self.logits[0].float()  # fp32 GPU，不搬 CPU

    def prefill(self, input_ids, cap_states=False):
        """input_ids → 最后 token logits[VOCAB] (fp32 GPU)。cap_states=True 写续接状态供 decode。"""
        ids = torch.tensor(input_ids, dtype=torch.int64, device=DEV)
        S = ids.numel()
        assert S <= self.max_tokens
        cur, other = self.hbuf[0], self.hbuf[1]
        B.reset_allocator()
        B.embedding(ptr(self.embed), ptr(ids), ptr(cur), S)
        for ly in self.layers:
            B.reset_allocator()
            if ly['type'] == 'mamba':
                self._run_mamba(ly, cur, other, S, cap_states)
            elif ly['type'] == 'attn':
                self._run_attn(ly, cur, other, S, cap_states)
            else:
                self._run_mlp(ly, cur, other, S)
            cur, other = other, cur
        self._S = S
        return self._lm_head_last(cur, S)

    def decode_step(self, token_id, s_cache):
        """单 token 续接。s_cache=已处理 token 数（新 token 落位 s_cache）。返回 logits[VOCAB]。"""
        ids = torch.tensor([token_id], dtype=torch.int64, device=DEV)
        cur, other = self.dbuf[0], self.dbuf[1]
        B.reset_allocator()
        B.embedding(ptr(self.embed), ptr(ids), ptr(cur), 1)
        for ly in self.layers:
            B.reset_allocator()
            if ly['type'] == 'mamba':
                self._dec_mamba(ly, cur, other)
            elif ly['type'] == 'attn':
                self._dec_attn(ly, cur, other, s_cache)
            else:
                self._run_mlp(ly, cur, other, 1)
            cur, other = other, cur
        return self._lm_head_last(cur, 1)

    @staticmethod
    def _sample(logits, temperature, top_p, recent_ids=None, rep_penalty=1.0):
        """温度 + nucleus(top_p) 采样 + 重复惩罚；temperature<=0 退化为贪心。
        logits: fp32 GPU [VOCAB]——全部算子在 GPU 上跑，结尾只 int() 回传 1 个 token id。"""
        if rep_penalty != 1.0 and recent_ids:
            idx = torch.tensor(sorted(set(recent_ids)), dtype=torch.long, device=logits.device)
            v = logits[idx]
            logits[idx] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
        if temperature <= 0.0:
            return int(logits.argmax())
        probs = torch.softmax(logits / temperature, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= top_p          # 保留累积概率刚好不超过 top_p 的核
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        sp /= sp.sum()
        return int(si[torch.multinomial(sp, 1)])

    def stream_generate(self, prompt_ids, max_new=1024, stop_ids=(11,),
                        temperature=0.6, top_p=0.95, rep_penalty=1.1, rep_window=64):
        """prefill(prompt) → 流式 decode（温度/top_p 采样 + 重复惩罚），逐 token yield id；
        遇 stop_ids 或 max_new 停。结束再 yield sentinel dict 含计时（'__stat__'）。
        temperature=0 → 贪心。
        ⚠️ 调参教训：temp=1.0 + 大窗口(256)重复惩罚会让 fp8 小模型长生成崩成乱码——
           被惩罚的不同 token 越积越多，常用字被压制，概率推向长尾垃圾。改 temp=0.6 +
           rep_penalty=1.1 + 仅近 64 token 窗口（只断局部循环，不压垮长程分布）。"""
        self.reset_states()
        stop = set(stop_ids)
        gen = []  # 已生成 token（重复惩罚用，取近 rep_window 个）
        B.sync(); t0 = time.time()
        lg = self.prefill(prompt_ids, cap_states=True)
        B.sync(); prefill_ms = (time.time() - t0) * 1000
        nxt = self._sample(lg, temperature, top_p, gen[-rep_window:], rep_penalty)
        pos = len(prompt_ids)
        n = 0
        t0 = time.time()
        while nxt not in stop and n < max_new and pos < self.max_tokens - 1:
            yield nxt
            gen.append(nxt)
            n += 1
            lg = self.decode_step(nxt, pos)
            nxt = self._sample(lg, temperature, top_p, gen[-rep_window:], rep_penalty)
            pos += 1
        dec_s = time.time() - t0
        yield {'__stat__': True, 'prefill_ms': prefill_ms,
               'decode_tps': n / dec_s if dec_s > 0 else 0.0, 'n': n}

    def generate(self, prompt_ids, n_new, greedy=True):
        """prefill(prompt) → 贪心 decode n_new token。返回 (生成 token 列表, prefill_ms, decode_ms_per_tok)。"""
        self.reset_states()
        import time
        B.sync(); t0 = time.time()
        lg = self.prefill(prompt_ids, cap_states=True)
        B.sync(); prefill_ms = (time.time() - t0) * 1000
        nxt = int(lg.argmax())
        out = [nxt]
        pos = len(prompt_ids)
        B.sync(); t0 = time.time()
        for _ in range(n_new - 1):
            lg = self.decode_step(nxt, pos)
            nxt = int(lg.argmax())
            out.append(nxt)
            pos += 1
        B.sync(); dec_ms = (time.time() - t0) * 1000 / max(n_new - 1, 1)
        return out, prefill_ms, dec_ms


if __name__ == '__main__':
    t0 = time.time()
    eng = NemotronEngine()
    print(f'加载完成 {time.time()-t0:.1f}s, layers={eng.nlayers}, '
          f'类型={[l["type"][0] for l in eng.layers]}')
    print('GPU 显存占用 %.2f GB' % (torch.cuda.memory_allocated()/1e9))

    ids = [1, 100, 200, 300, 400, 500, 42, 17, 9999, 2]
    # 预热（首调含 cuBLAS handle/plan 初始化）
    eng.generate(ids, 4)
    # 实测
    gen, pf_ms, dec_ms = eng.generate(ids, 33)
    print(f'prefill {len(ids)} tok: {pf_ms:.1f}ms')
    print(f'decode: {dec_ms:.2f}ms/tok  →  {1000/dec_ms:.1f} tok/s')
    print('生成 token:', gen[:16], '...')

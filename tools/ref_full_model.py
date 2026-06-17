"""
ref_full_model.py — Nemotron-H 整模型纯 torch 参考（CPU fp32），产出参考 logits。

复用已逐块对齐 HF 的数学（dump_{mamba,attention,mlp}_block.py verbatim）：
  - Mamba2: in_proj→conv1d+SiLU→chunked SSD(+D)→gated RMSNorm→out_proj
  - Attention: NoPE GQA causal SDPA
  - MLP: up→relu²→down
  - 外层 pre-norm 残差；embed / final norm / lm_head

权重从 safetensors 加载，fp8 dequant=w*scale(标量)。CPU fp32 跑，避免 8GB 显存墙与 transformers 依赖。
输出 tests/data/full_model/logits_ref.bin + meta（固定输入 token）。

跑法：uv run python tools/ref_full_model.py
"""
import os, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)
MD = os.path.join(os.path.dirname(__file__), '..', 'model', 'NVIDIA-Nemotron-3-Nano-4B-FP8')
OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'full_model')
os.makedirs(OUT, exist_ok=True)
F32 = torch.float32

cfg = json.load(open(os.path.join(MD, 'config.json')))
HIDDEN = cfg['hidden_size']; EPS = cfg['layer_norm_epsilon']
H = cfg['mamba_num_heads']; P = cfg['mamba_head_dim']; N = cfg['ssm_state_size']; G = cfg['n_groups']
CONV_K = cfg['conv_kernel']; CHUNK = cfg['chunk_size']
H_Q = cfg['num_attention_heads']; H_KV = cfg['num_key_value_heads']; HEAD = cfg['attention_head_dim']
INTER_MLP = cfg['intermediate_size']
INTER = H * P; CONV_DIM = INTER + 2 * G * N
GROUP_SIZE = INTER // G
DT_MIN, DT_MAX = 0.0, float('inf')

# ---- 加载权重（CPU fp32）----
raw = {}
for fn in glob.glob(os.path.join(MD, '*.safetensors')):
    with safe_open(fn, 'pt') as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

def W(name):
    w = raw[name + '.weight']
    if w.dtype == torch.float8_e4m3fn:
        s = raw[name + '.weight_scale'].float().reshape(())
        return w.float() * s
    return w.float()
def T(name):
    return raw[name].float()

nlayers = 0
while f'backbone.layers.{nlayers}.norm.weight' in raw:
    nlayers += 1

# ---- 数学（verbatim from dump scripts）----
def pad_by(t, n):
    sh = (0,0,0,0,0,n,0,0) if t.dim()==4 else (0,0,0,n,0,0)
    return F.pad(t, sh)
def reshape_chunks(t, pad, c):
    t = pad_by(t, pad)
    if t.dim()==3: return t.reshape(t.shape[0], -1, c, t.shape[2])
    return t.reshape(t.shape[0], -1, c, t.shape[2], t.shape[3])
def segsum(t):
    c = t.size(-1)
    t = t[..., None].expand(*t.size(), c)
    m = torch.tril(torch.ones(c, c, dtype=torch.bool), -1)
    t = t.masked_fill(~m, 0)
    s = torch.cumsum(t, dim=-2)
    m = torch.tril(torch.ones(c, c, dtype=torch.bool), 0)
    return s.masked_fill(~m, -torch.inf)
def rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
def gated_rms(x, w, z, gs):
    x = x.float() * F.silu(z.float())
    xg = x.reshape(*x.shape[:-1], x.shape[-1]//gs, gs)
    r = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + EPS)
    return (xg*r).reshape(x.shape) * w

def mamba(x, p):
    Bsz, S, _ = x.shape
    proj = F.linear(x, W(p+'mixer.in_proj'))
    gate, xbc, dt = torch.split(proj, [INTER, CONV_DIM, H], dim=-1)
    cw = T(p+'mixer.conv1d.weight'); cb = T(p+'mixer.conv1d.bias')
    h = xbc.transpose(1,2)
    h = F.conv1d(h, cw, bias=cb, padding=CONV_K-1, groups=CONV_DIM)[..., :S]
    h = F.silu(h.transpose(1,2))
    xs, Bm, Cm = torch.split(h, [INTER, G*N, G*N], dim=-1)
    dt = torch.clamp(F.softplus(dt + T(p+'mixer.dt_bias')), DT_MIN, DT_MAX)
    hs = xs.reshape(Bsz, S, -1, P).float()
    Bm = Bm.reshape(Bsz, S, -1, N).float().repeat_interleave(H//G, dim=2)
    Cm = Cm.reshape(Bsz, S, -1, N).float().repeat_interleave(H//G, dim=2)
    pad = (CHUNK - S % CHUNK) % CHUNK
    A = -torch.exp(T(p+'mixer.A_log').float())
    Dres = T(p+'mixer.D')[..., None] * pad_by(hs, pad)
    hs = hs * dt[..., None]
    A = A * dt
    hs, A, Bm, Cm = [reshape_chunks(t, pad, CHUNK) for t in (hs, A, Bm, Cm)]
    A = A.permute(0,3,1,2); Acs = torch.cumsum(A, dim=-1)
    Lm = torch.exp(segsum(A))
    Gi = (Cm[:,:,:,None,:,:] * Bm[:,:,None,:,:,:]).sum(-1)
    M = (Gi[...,None] * Lm.permute(0,2,3,4,1)[...,None]).sum(-1)
    Ydiag = (M[...,None] * hs[:,:,None]).sum(3)
    decay = torch.exp(Acs[:,:,:,-1:] - Acs)
    Bd = Bm * decay.permute(0,-2,-1,1)[...,None]
    st = (Bd[...,None,:] * hs[...,None]).sum(2)
    st = torch.cat([torch.zeros_like(st[:,:1]), st], dim=1)
    dchunk = torch.exp(segsum(F.pad(Acs[:,:,:,-1], (1,0)))).transpose(1,3)
    nst = (dchunk[...,None,None] * st[:,:,None,...]).sum(1)
    st = nst[:,:-1]
    sdo = torch.exp(Acs)
    Yoff = (Cm[...,None,:] * st[:,:,None,...]).sum(-1) * sdo.permute(0,2,3,1)[...,None]
    y = (Ydiag + Yoff).reshape(Bsz, -1, H, P) + Dres
    if pad>0: y = y[:, :S]
    y = y.reshape(Bsz, S, -1)
    y = gated_rms(y, T(p+'mixer.norm.weight'), gate, GROUP_SIZE)
    return F.linear(y, W(p+'mixer.out_proj'))

def repeat_kv(h, r):
    b,hh,s,d = h.shape
    if r==1: return h
    return h[:,:,None,:,:].expand(b,hh,r,s,d).reshape(b,hh*r,s,d)

def attn(x, p):
    Bsz, S, _ = x.shape
    q = F.linear(x, W(p+'mixer.q_proj')).view(Bsz,S,H_Q,HEAD).transpose(1,2)
    k = F.linear(x, W(p+'mixer.k_proj')).view(Bsz,S,H_KV,HEAD).transpose(1,2)
    v = F.linear(x, W(p+'mixer.v_proj')).view(Bsz,S,H_KV,HEAD).transpose(1,2)
    k = repeat_kv(k, H_Q//H_KV); v = repeat_kv(v, H_Q//H_KV)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # NoPE
    o = o.transpose(1,2).contiguous().view(Bsz,S,H_Q*HEAD)
    return F.linear(o, W(p+'mixer.o_proj'))

def mlp(x, p):
    return F.linear(F.relu(F.linear(x, W(p+'mixer.up_proj'))).pow(2), W(p+'mixer.down_proj'))

def forward(ids):
    x = F.embedding(ids, T('backbone.embeddings.weight'))[None]  # [1,S,H]
    for i in range(nlayers):
        p = f'backbone.layers.{i}.'
        res = x
        h = rms(x, T(p+'norm.weight'))
        if p+'mixer.in_proj.weight' in raw:   mix = mamba(h, p)
        elif p+'mixer.q_proj.weight' in raw:  mix = attn(h, p)
        else:                                  mix = mlp(h, p)
        x = res + mix
    x = rms(x, T('backbone.norm_f.weight'))
    return F.linear(x, T('lm_head.weight'))[0]  # [S, VOCAB]

if __name__ == '__main__':
    ids = torch.tensor([1, 100, 200, 300, 400, 500, 42, 17, 9999, 2], dtype=torch.long)
    print(f'nlayers={nlayers}, 跑 CPU fp32 参考（{ids.numel()} tok）...')
    with torch.no_grad():
        logits = forward(ids)  # [S, VOCAB]
    last = logits[-1].contiguous()
    last.numpy().astype('<f4').tofile(os.path.join(OUT, 'logits_ref.bin'))
    ids.numpy().astype('<i8').tofile(os.path.join(OUT, 'input_ids.bin'))
    with open(os.path.join(OUT, 'meta.txt'), 'w') as f:
        f.write(f'S={ids.numel()} VOCAB={logits.shape[-1]}\n')
    top = torch.topk(last, 5)
    print('参考 top5 token:', top.indices.tolist())
    print('参考 top5 logit:', [round(x,3) for x in top.values.tolist()])
    print('参考 argmax =', int(last.argmax()))
    print('saved to', os.path.abspath(OUT))

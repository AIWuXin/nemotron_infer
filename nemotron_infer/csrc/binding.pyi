"""nemotron_infer bf16 whole-model orchestration bindings"""
from __future__ import annotations
import nemotron_infer.csrc.binding
import typing

__all__ = [
    "CONV_DIM",
    "CONV_K",
    "G",
    "H",
    "HEAD",
    "HIDDEN",
    "H_KV",
    "H_Q",
    "INTER",
    "IN_PROJ",
    "N",
    "P",
    "VOCAB",
    "attn_decode",
    "attn_decode_fp8",
    "attn_forward",
    "attn_forward_fp8",
    "embedding",
    "free_allocator",
    "lm_head",
    "mamba_decode",
    "mamba_decode_fp8",
    "mamba_forward",
    "mamba_forward_fp8",
    "mlp_forward",
    "mlp_forward_fp8",
    "reset_allocator",
    "rmsnorm",
    "sync"
]


def attn_decode(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def attn_decode_fp8(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex, arg11: typing.SupportsInt | typing.SupportsIndex, arg12: typing.SupportsInt | typing.SupportsIndex, arg13: typing.SupportsInt | typing.SupportsIndex, arg14: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def attn_forward(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def attn_forward_fp8(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex, arg11: typing.SupportsInt | typing.SupportsIndex, arg12: typing.SupportsInt | typing.SupportsIndex, arg13: typing.SupportsInt | typing.SupportsIndex, arg14: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def embedding(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def free_allocator() -> None:
    pass
def lm_head(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def mamba_decode(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex, arg11: typing.SupportsInt | typing.SupportsIndex, arg12: typing.SupportsInt | typing.SupportsIndex, arg13: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def mamba_decode_fp8(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex, arg11: typing.SupportsInt | typing.SupportsIndex, arg12: typing.SupportsInt | typing.SupportsIndex, arg13: typing.SupportsInt | typing.SupportsIndex, arg14: typing.SupportsInt | typing.SupportsIndex, arg15: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def mamba_forward(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex, arg11: typing.SupportsInt | typing.SupportsIndex, arg12: typing.SupportsInt | typing.SupportsIndex, arg13: typing.SupportsInt | typing.SupportsIndex, arg14: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def mamba_forward_fp8(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex, arg8: typing.SupportsInt | typing.SupportsIndex, arg9: typing.SupportsInt | typing.SupportsIndex, arg10: typing.SupportsInt | typing.SupportsIndex, arg11: typing.SupportsInt | typing.SupportsIndex, arg12: typing.SupportsInt | typing.SupportsIndex, arg13: typing.SupportsInt | typing.SupportsIndex, arg14: typing.SupportsInt | typing.SupportsIndex, arg15: typing.SupportsInt | typing.SupportsIndex, arg16: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def mlp_forward(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def mlp_forward_fp8(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.SupportsInt | typing.SupportsIndex, arg5: typing.SupportsInt | typing.SupportsIndex, arg6: typing.SupportsInt | typing.SupportsIndex, arg7: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def reset_allocator() -> None:
    pass
def rmsnorm(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def sync() -> None:
    pass
CONV_DIM = 9728
CONV_K = 4
G = 8
H = 96
HEAD = 128
HIDDEN = 3136
H_KV = 8
H_Q = 40
INTER = 12544
IN_PROJ = 17504
N = 128
P = 80
VOCAB = 131072

"""nemotron_infer — Nemotron-3-Nano-4B-FP8 手写 CUDA 推理引擎。

公共入口：

    from nemotron_infer import NemotronEngine
    eng = NemotronEngine()
    tokens, prefill_ms, dec_ms = eng.generate(prompt_ids, n_new=128)

交互式终端：python -m nemotron_infer.chat
"""
from nemotron_infer.engine import NemotronEngine, MODEL_DIR

__all__ = ["NemotronEngine", "MODEL_DIR"]
__version__ = "0.1.0"

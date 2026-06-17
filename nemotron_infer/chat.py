"""
chat.py — Nemotron-3-Nano-4B-FP8 交互式终端（手写 CUDA 引擎驱动）。

ChatML reasoning 格式：prompt 以 `<|im_start|>assistant\n<think>\n` 收尾，模型先输出思维链，
`</think>` 后是回答，`<|im_end|>` 停止。终端用 rich 流式渲染：思维链(暗色) + 回答(Markdown)。

跑法：uv run python -m nemotron_infer.chat
命令：/exit 退出  /clear 清空对话  /system <内容> 设系统提示
"""
import os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
from nemotron_infer.engine import NemotronEngine, MODEL_DIR
import torch
from tokenizers import Tokenizer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

# 特殊 token
IM_START, IM_END, THINK_O, THINK_C = 10, 11, 12, 13

console = Console()


def build_prompt(messages, system=""):
    """ChatML：始终发 system 块（可空）→ 历史 → assistant 生成提示(开思维链)。"""
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for m in messages:
        if m['role'] == 'assistant':
            s += f"<|im_start|>assistant\n<think></think>{m['content']}<|im_end|>\n"
        else:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    s += "<|im_start|>assistant\n<think>\n"
    return s


def banner(eng):
    vram = torch.cuda.memory_allocated() / 1e9
    name = torch.cuda.get_device_name(0)
    nmamba = sum(1 for l in eng.layers if l['type'] == 'mamba')
    nattn = sum(1 for l in eng.layers if l['type'] == 'attn')
    nmlp = sum(1 for l in eng.layers if l['type'] == 'mlp')
    info = Text.assemble(
        ("  Nemotron-3-Nano-4B-FP8  ", "bold white on dark_cyan"),
        ("  手写 CUDA 推理引擎\n", "bold cyan"),
        (f"  {eng.nlayers} 层  ", "dim"),
        (f"{nmamba} Mamba2 / {nattn} Attention / {nmlp} MLP", "cyan"),
        (f"   ·   {vram:.2f} GB 显存   ·   {name}\n", "dim"),
        ("  /exit 退出   /clear 清空   /system <内容> 设系统提示", "dim italic"),
    )
    console.print(Panel(info, box=box.ROUNDED, border_style="dark_cyan", padding=(0, 1)))


def _emit(seg_ids, prev_len, tk, style):
    """追加式打印：解码整段，仅输出新增后缀。
    ⚠️ byte-level BPE：多字节 UTF-8 字符跨 token 时，未完成字节解码成 U+FFFD(�)。
       必须剥掉结尾的 � 只打印稳定前缀，等后续 token 补全该字符后再输出，否则乱码。"""
    full = tk.decode(seg_ids, skip_special_tokens=True)
    stable = full.rstrip('�')          # 结尾未完成字符暂不输出
    delta = stable[prev_len:]
    if delta:
        console.print(delta, end="", style=style, markup=False, highlight=False, soft_wrap=True)
    return len(stable)


def chat_once(eng, tk, messages, system, max_new=2048):
    """追加式流式（永不重刷，任意长度不刷屏）：思维链暗色 + 回答正常色。"""
    prompt = build_prompt(messages, system)
    ids = tk.encode(prompt).ids
    if len(ids) >= eng.max_tokens - max_new:
        max_new = max(64, eng.max_tokens - len(ids) - 1)

    think_ids, ans_ids, mode = [], [], 'think'
    think_prev, ans_prev = 0, 0
    th_hdr = an_hdr = False
    stat = None

    for tok in eng.stream_generate(ids, max_new=max_new, stop_ids=(IM_END,)):
        if isinstance(tok, dict):
            stat = tok
            break
        if tok == THINK_C:
            mode = 'answer'
            continue
        if tok == THINK_O:
            continue
        if mode == 'think':
            if not th_hdr:
                console.print(Rule("🤔 思考过程", style="grey42", characters="·"))
                th_hdr = True
            think_ids.append(tok)
            think_prev = _emit(think_ids, think_prev, tk, "grey58")
        else:
            if not an_hdr:
                console.print()
                console.print(Rule("💬 回答", style="green"))
                an_hdr = True
            ans_ids.append(tok)
            ans_prev = _emit(ans_ids, ans_prev, tk, "white")

    console.print()  # 收尾换行
    ans = tk.decode(ans_ids, skip_special_tokens=True).strip()
    think = tk.decode(think_ids, skip_special_tokens=True).strip()
    if not ans and not think:
        console.print("[dim](模型未生成内容)[/dim]")
    if stat:
        console.print(f"[dim]  prefill {len(ids)} tok / {stat['prefill_ms']:.0f}ms   ·   "
                      f"decode {stat['n']} tok @ {stat['decode_tps']:.1f} tok/s[/dim]")
    return ans


def main():
    console.print("[dim]加载引擎中…[/dim]")
    t0 = time.time()
    eng = NemotronEngine(max_tokens=4096)
    tk = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    console.print(f"[dim]就绪 ({time.time()-t0:.1f}s)[/dim]\n")
    banner(eng)

    system = ""
    messages = []
    while True:
        try:
            user = console.input("\n[bold cyan]你 ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见。[/dim]")
            break
        if not user:
            continue
        if user in ('/exit', '/quit'):
            console.print("[dim]再见。[/dim]")
            break
        if user == '/clear':
            messages = []
            console.print("[dim]对话已清空。[/dim]")
            continue
        if user.startswith('/system'):
            system = user[len('/system'):].strip()
            messages = []
            console.print(f"[dim]系统提示已设置：{system!r}[/dim]")
            continue

        messages.append({'role': 'user', 'content': user})
        try:
            answer = chat_once(eng, tk, messages, system)
            if answer:
                messages.append({'role': 'assistant', 'content': answer})
            else:
                messages.pop()  # 空回复不入历史，避免污染后续对话
        except KeyboardInterrupt:
            console.print("\n[yellow]已中断生成。[/yellow]")
            messages.pop()


if __name__ == '__main__':
    main()

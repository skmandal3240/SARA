#!/usr/bin/env python3
"""Build a tiny public-domain + synthetic multimodal corpus and train the BPE tokenizer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from sara.config import SARAConfig, load_config
from sara.tokenizer import SARATokenizer, SPECIAL_TOKENS
from sara.vision import make_shape_image, pil_to_tensor

ROOT = Path(__file__).resolve().parent

# Public-domain / original short text (not copied from copyrighted books).
TEXT = [
    "The sun is a star at the center of the solar system.",
    "Water freezes at zero degrees Celsius and boils at one hundred.",
    "A triangle has three sides and three angles.",
    "Python is a programming language used for science and tools.",
    "Agents plan, call tools, observe results, and try again after errors.",
    "SARA can see images, talk in speech, write code, and compose songs.",
    "Red circles sit on dark canvases. Blue squares do too.",
    "Fibonacci numbers start 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55.",
    "The tenth Fibonacci number is 55 when counting F(0)=0 and F(10)=55.",
    "Rhythm needs a beat. Harmony needs a chord. Melody needs a scale.",
    "A tool call is a structured action, not a guess in prose.",
    "Observation comes back into context so the next thought can use it.",
    "Write a file, run it, read the stdout, then answer.",
    "The critic checks the coder. The orchestrator delegates the work.",
    "Speech is a waveform. We turn it into a log-mel spectrogram.",
    "Griffin-Lim estimates phase so a spectrogram can become a wav file.",
    "Cross-attention lets language queries look at visual memory tokens.",
    "Grouped query attention shares keys and values across head groups.",
    "Rotary embeddings encode position as a rotation in each pair of dims.",
    "RMSNorm scales hidden states by their root-mean-square.",
]

CODE = [
    "def add(a, b):\n    return a + b\nprint(add(2, 3))\n",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(10))\n",
    "print(sum(range(1, 11)))\n",
    "xs = [1, 2, 3, 4]\nprint(xs[-1] * xs[0])\n",
    "def fact(n):\n    p = 1\n    for i in range(1, n + 1):\n        p *= i\n    return p\nprint(fact(6))\n",
]

TOOL_TRACES = [
    (
        "What is 17 times 3?",
        '<|thought|>I should use the calculator.\n'
        '<|tool_call|>{"name": "calc", "args": {"expr": "17*3"}}<|tool_end|>\n'
        '<|observation|>{"ok": true, "result": {"value": 51}}\n'
        "<|final|>51",
    ),
    (
        "Write a python file that prints the 10th fibonacci number, run it, report the result.",
        '<|thought|>Write the file, then run it with python3.\n'
        '<|tool_call|>{"name": "file_write", "args": {"path": "outputs/agent_task.py", '
        '"content": "def fib(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a\\nprint(fib(10))\\n"}}'
        "<|tool_end|>\n"
        '<|observation|>{"ok": true, "result": {"path": "outputs/agent_task.py"}}\n'
        '<|tool_call|>{"name": "shell", "args": {"command": "python3 outputs/agent_task.py"}}<|tool_end|>\n'
        '<|observation|>{"ok": true, "result": {"returncode": 0, "stdout": "55\\n"}}\n'
        "<|final|>55",
    ),
    (
        "What time is it?",
        '<|thought|>Use the now tool.\n'
        '<|tool_call|>{"name": "now", "args": {}}<|tool_end|>\n'
        "<|final|>The clock tool returned UTC and IST timestamps.",
    ),
]


def build_sequences(tok: SARATokenizer) -> list[list[int]]:
    seqs: list[list[int]] = []
    for t in TEXT:
        s = f"<|bos|><|user|>{t}<|assistant|>{t}<|eos|>"
        seqs.append(tok.encode(s))
    for c in CODE:
        s = f"<|bos|><|user|>write a small program<|code_start|>{c}<|code_end|><|eos|>"
        seqs.append(tok.encode(s))
    for goal, trace in TOOL_TRACES:
        s = f"<|bos|><|user|>{goal}<|assistant|>{trace}<|eos|>"
        seqs.append(tok.encode(s))
    colors = ["red", "green", "blue", "yellow", "purple"]
    kinds = ["circle", "square", "triangle"]
    for color in colors:
        for kind in kinds:
            cap = f"a {color} {kind}"
            s = f"<|bos|><|user|>describe this image<|assistant|>{cap}<|eos|>"
            seqs.append(tok.encode(s))
            s = f"<|bos|><|user|>{cap}<|img_gen|><|eos|>"
            seqs.append(tok.encode(s))
            s = f"<|bos|><|user|>make a short video of {cap}<|vid_gen|><|eos|>"
            seqs.append(tok.encode(s))
            s = f"<|bos|><|user|>compose a song about {color} {kind}s<|song_gen|><|eos|>"
            seqs.append(tok.encode(s))
    # speech / talk
    for phrase in ["hello sara", "a red circle", "print fibonacci"]:
        s = f"<|bos|><|speech_in|>{phrase}<|assistant|>{phrase}<|eos|>"
        seqs.append(tok.encode(s))
        s = f"<|bos|><|user|>say {phrase}<|speech_out|>{phrase}<|eos|>"
        seqs.append(tok.encode(s))
    return seqs


def pack_bin(seqs: list[list[int]], path: Path, seq_len: int, pad_id: int) -> int:
    rng = random.Random(0)
    chunks = []
    for s in seqs:
        if len(s) > seq_len:
            s = s[:seq_len]
        if len(s) < 4:
            continue
        padded = s + [pad_id] * (seq_len - len(s))
        chunks.append(padded)
    # repeat to have enough steps
    while len(chunks) < 256:
        chunks.append(rng.choice(chunks)[:])
    arr = np.array(chunks, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return len(arr)


def write_shapes(out_dir: Path, img_size: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for color in ["red", "green", "blue", "yellow"]:
        for kind in ["circle", "square", "triangle"]:
            img = make_shape_image(kind, color, size=img_size)
            name = f"{color}_{kind}.png"
            img.save(out_dir / name)
            meta.append({"file": name, "caption": f"a {color} {kind}", "kind": kind, "color": color})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "sara_nano.yaml"))
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument(
        "--source",
        default=None,
        help="hf:org/name or catalog id; streams a tiny sample via sara.data (never vendors TB)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    vocab = args.vocab_size or cfg.vocab_size

    texts = list(TEXT) + CODE + [g + "\n" + t for g, t in TOOL_TRACES]
    texts += SPECIAL_TOKENS

    if getattr(args, "source", None):
        from sara.data.adapters import ingest_source

        extra = ingest_source(args.source, max_tokens=2048)
        texts.extend(extra)
        print(f"source {args.source}: +{len(extra)} docs (capped)")
    texts += [
        "file_write shell python_exec calc now web_search list_files image_gen audio_gen code_run",
        "<|tool_call|>{\"name\": \"calc\", \"args\": {\"expr\": \"1+1\"}}<|tool_end|>",
    ]
    tok_path = ROOT / "tokenizer" / "sara.json"
    print(f"training BPE vocab={vocab} on {len(texts)} docs → {tok_path}")
    tok = SARATokenizer.train(texts, vocab_size=vocab, save_path=tok_path)
    print(f"tokenizer vocab_size={tok.vocab_size} pad={tok.pad_id} eos={tok.eos_id}")

    seqs = build_sequences(tok)
    n = pack_bin(seqs, ROOT / "data" / "train.bin", cfg.max_seq_len, tok.pad_id)
    print(f"wrote data/train.bin sequences={n} seq_len={cfg.max_seq_len}")

    meta = write_shapes(ROOT / "data" / "shapes", cfg.img_size)
    print(f"wrote {len(meta)} synthetic shapes under data/shapes/")

    # a tiny wav of mixed tones for listen tests
    sr = cfg.sample_rate
    t = np.arange(int(sr * 0.6), dtype=np.float32) / sr
    wav = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)
    try:
        from sara.audio import save_wav

        save_wav(str(ROOT / "data" / "tone.wav"), wav, sr)
        print("wrote data/tone.wav")
    except Exception as e:
        print("wav skip", e)


if __name__ == "__main__":
    main()

# SARA

**See, Articulate, Reason, Author** — a *from-scratch* multimodal transformer with a first-class **tool-using agent runtime**.

SARA is not an API wrapper. Attention, RMSNorm, RoPE, grouped-query attention, SwiGLU, modality encoders/decoders, the train loop, generate loops, the tool protocol, and the multi-agent loop are implemented in this repo (PyTorch).

Nano weights are a **proof of the runtime**. The architecture and agent stack are what scale; a few hundred CPU steps will not make a frontier model.

## What it can do

| Pillar | Path | Honest nano status |
|---|---|---|
| **SEE** | Image → patch encoder → transformer (cross-attn + caption generate) | Encoder/decoder run; captions are weakly trained |
| **TALK** | Text chat; wav→log-mel→tokens (listen); hidden→mel→Griffin-Lim wav (speak) | Real wav I/O; speech is not a production ASR/TTS |
| **CODE** | `<\|code_start\|>` pathway, `ast.parse`, sandboxed `python_exec` | Demo writes, syntax-checks, and runs a program |
| **CREATE IMAGE** | Pooled hidden → conv-transpose RGB (64×64) | Trained on synthetic colored shapes |
| **CREATE VIDEO** | Time-conditioned frame decoder → GIF | Short 8-frame GIF, low res |
| **CREATE SONG** | SongHead (key/tempo/scale/notes) + additive synth (melody+bass+pad+hats) | Not a sine beep; not a studio model |
| **TOOLS / AGENTS** | JSON/`<\|tool_call\|>` protocol, ToolHead, ReAct loop, planner, swarm | Runtime is real; nano demo writes+runs `fib.py` and reports 55. Weights are still nano. |

## Architecture glance

```
tokens + modality-type embeddings
        │
        ▼
  ×6 Transformer blocks   GQA (4Q / 2KV) · RoPE · QK-RMSNorm · SwiGLU
        │    cross-attn every other layer ← vision tokens / audio tokens
        ▼
   RMSNorm · tied LM head
        │
        ├── ImageDecoder (conv-T)     ├── MelDecoder + Griffin-Lim
        ├── VideoDecoder (per-frame)  ├── SongHead → additive synth
        └── ToolHead (tool id + stop)
```

- **Unified stream:** text, code, and special tokens share one BPE (Hugging Face `tokenizers`).
- **Vision:** 8×8 patches on 64×64 RGB, tiny bidirectional encoder, concat/cross-attn into language.
- **Audio:** log-mel filterbank, 1D conv encoder, Griffin-Lim vocoder (no external neural vocoder).
- **Music:** discrete score from the transformer, rendered with harmonics, chords, kick/hat — not one oscillator.
- **Agents:** see below.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # CPU torch is fine

python prepare_data.py                   # BPE + tiny corpus + synthetic shapes
python train.py --steps 300              # ~minutes on CPU
python -m pytest tests/ -q
python demos.py                          # artifacts → outputs/  (exit 0)
```

Generate text:

```bash
python generate.py --prompt "The sun is a star"
python generate.py --code --prompt "print the 10th fibonacci number"
python tools.py catalog
python tools.py agent --goal "What is 17*3?"
```

Checkpoint path: `checkpoints/sara_nano/sara.pt` (gitignored). Tokenizer: `tokenizer/sara.json`.

## Agents and tools

Tool use is a **core pillar**, not a print stub.

### Protocol

The model is trained to emit:

```
<|thought|>I should calculate it.
<|tool_call|>{"name": "calc", "args": {"expr": "17*3"}}<|tool_end|>
<|observation|>{"ok": true, "result": {"value": 51}}
<|final|>51
```

Alternate parse: `[[tool:calc(expr=17*3)]]`. A **ToolHead** on the pooled hidden state classifies tool-id vs stop for constrained decode.

### Loop

`sara.agent.loop.AgentRuntime`:

1. Build a plan (`sara.agent.planner` — model plan if parseable, else a task-shape skeleton).
2. ReAct: thought → tool call → observation back into the transformer context.
3. On tool error, keep the failed step pending, inject the error, retry (e.g. rewrite the file after a `SyntaxError`).
4. Stop on `<|final|>`, ToolHead stop, empty pending plan, `max_steps`, or `max_tool_calls`.

Memory: in-context **scratchpad** + file-backed **long-term notes** (`outputs/sara_memory.json`).

### Built-in tools

| Name | What it actually does |
|---|---|
| `python_exec` | AST-guarded sandbox (blocked `os`/`subprocess`/…), captures stdout |
| `code_run` | `ast.parse` then sandbox exec |
| `calc` | Arithmetic AST evaluator |
| `now` | UTC + IST timestamps |
| `file_read` / `file_write` / `list_files` | Jailed to the workspace root |
| `shell` | Allowlist: `python3`, `ls`, `cat`, `wc`, `date`, `echo`, `pwd` |
| `web_search` | DuckDuckGo HTML fetch; returns `{ok:false, reason:...}` if offline |
| `image_gen` / `audio_gen` | Call SARA's own image/song heads when a runtime is bound |

### Add a tool

```python
from sara.agent.loop import AgentRuntime
from sara.tools.registry import ToolSpec

rt = AgentRuntime("/path/to/workspace", model=model, tokenizer=tok, cfg=cfg)
rt.registry.add(
    "reverse",
    lambda text: text[::-1],
    "Reverse a string.",
    {"text": "str"},
    ["text"],
)
```

### Multi-agent swarm

`sara.agent.swarm.Swarm`: **orchestrator** delegates to **coder** / **researcher**; **critic** can `REJECT` and send work back. Nano only runs a handful of steps; the objects (roles, delegation records, shared long-term memory) are the scalable API.

```python
from sara.agent import AgentRuntime, Swarm
rt = AgentRuntime(".", model=model, tokenizer=tok, cfg=cfg)
print(Swarm(rt).orchestrate("Write a python program that prints factorial of 6.").final)
```

## Layout

```
sara/                 package (config, modules, model, tokenizer, vision, audio, video, music)
  tools/              protocol, registry, builtins
  agent/              loop, planner, swarm, memory
  edge/               profiles, runtime, mesh, INT8, paging
  privacy/            grants, vault, audit, local learn
  data/               dataset catalog + HF adapters (no corpora)
configs/              sara_nano.yaml + edge_*.yaml
prepare_data.py  train.py  generate.py  demos.py  tools.py
tests/           outputs/  data/  tokenizer/  checkpoints/
```

## Edge / company

SARA is meant to run **on the chip in the device**, not as a cloud API. Nano weights are a **runtime proof**, not a frontier model. Device profiles, INT8, layer paging, in-process mesh, and the ASTRO-style grant kernel are in this repo. The plan that does not fail: [`docs/PLAN.md`](docs/PLAN.md).

```bash
python -m pytest tests/test_edge.py tests/test_privacy.py tests/test_data_catalog.py tests/test_tools.py -q
python demos_edge.py    # cctv see-path, phone agent, 2-peer mesh — cloud denied; exit 0
```

- [`docs/ISA.md`](docs/ISA.md) — SARA-ISA ops (GQA, RoPE, RMSNorm, SwiGLU, conv-T, mel) and CPU → INT8 → FPGA → 28nm ASIC
- [`docs/GRANTS.md`](docs/GRANTS.md) — TIDE 2.0 / IndiaAI / DLI checklist; **DPIIT is the grant gate**; Pvt Ltd is a blocker
- [`docs/DATASETS.md`](docs/DATASETS.md) — adapters only (`sara.data`); never vendor TB into git

## Train notes

- Optimizer: AdamW, cosine LR, grad accum, fp32 CPU.
- Losses: next-token CE + MSE on synthetic shape images (see/create) + light song/mel regularizers.
- Do not commit `*.pt` or `data/*.bin`. Re-run `prepare_data.py` then `train.py`.

## What's next

- Larger vocab and a real web/code mix (still from-scratch train, not an API).
- Discrete VQ image tokens so CREATE and SEE share a codebook.
- A learned vocoder instead of Griffin-Lim.
- Tighter constrained decoding for tool JSON (FSM / grammar).
- Longer-horizon swarm with parallel workers.

## License

[Apache License 2.0](LICENSE). Training text is short original/public-domain sentences plus synthetic programs and shapes — no scraped books.

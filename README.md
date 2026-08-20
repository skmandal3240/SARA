# SARA

A from-scratch multimodal transformer with a first-class tool-using agent runtime.

**Stack:** PyTorch (CPU). **License:** Apache-2.0. **Status:** PoC.

[SARA — See, Articulate, Reason, Author](#what-sara-is) · [Quickstart](#quickstart) · [Architecture](#architecture) · [Agents and tools](#agents-and-tools) · [Edge runtime](#edge-runtime) · [Tests and demos](#tests-and-demos) · [Layout](#layout) · [Docs](#docs) · [Why this exists](#why-this-exists)

## What SARA is

SARA is a small multimodal transformer (10.8M parameters) and the agent runtime that runs on top of it. The transformer, tool protocol, sandbox, planner, and swarm are all implemented in this repo. Nothing calls OpenAI, Anthropic, or any other hosted model.

The model drives five input/output modalities:

- **Text** — generation, instruction following
- **Image** — patch encoder → caption decode
- **Audio** — log-mel encoder (listen), Griffin-Lim vocoder (speak)
- **Code** — `<|code_start|>` pathway, abstract-syntax-tree sandboxed execution
- **Song** — discrete score head → additive synthesis (melody, bass, pad, hats)

The agent runtime is a separate layer that uses the model to decide what to do, then calls tools, observes results, and continues until the task is done or a stop condition is met.

Nano weights are a runtime proof. A few hundred CPU training steps will not produce a frontier model. The architecture, tool protocol, and edge runtime are what scale.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py     # tokenizer + tiny corpus + synthetic shapes
python train.py --steps 50 # ~15 s on CPU
python -m pytest tests/ -q
python demos.py            # 9 demos, all exit 0
python demos_edge.py       # 3 edge demos, all exit 0
```

Generate text:

```bash
python generate.py --prompt "The sun is a star"
python generate.py --code --prompt "print the 10th fibonacci number"
```

Run the agent:

```bash
python tools.py catalog
python tools.py agent --goal "What is 17*3?"
```

## Architecture

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

- **Backbone:** 6 transformer blocks, 256-dim, 10.8M params, GQA, RoPE, RMSNorm, SwiGLU, tied LM head.
- **Modality encoders:** 8×8 patch encoder for 64×64 images; 1D conv over log-mel for audio.
- **Modality decoders:** conv-transpose image decoder (64×64); Griffin-Lim mel vocoder; per-frame video decoder; SongHead.
- **Tokens:** unified BPE over text, code, and control tokens (`tokenizer/sara.json`).
- **No external neural vocoder.** Griffin-Lim is enough for nano and runs anywhere.

## Agents and tools

Tool use is a core pillar, not a print stub.

### Protocol

The model emits:

```
<|thought|>I should calculate it.
<|tool_call|>{"name": "calc", "args": {"expr": "17*3"}}<|tool_end|>
<|observation|>{"ok": true, "result": {"value": 51}}
<|final|>51
```

A parsed alternate form `[[tool:calc(expr=17*3)]]` is also accepted. A **ToolHead** classifies the next token as `tool-id` or `stop` for constrained decode.

### ReAct loop

`sara.agent.loop.AgentRuntime`:

1. Build a plan (`sara.agent.planner` — model plan if parseable, else a task-shape skeleton).
2. ReAct: thought → tool call → observation back into the transformer context.
3. On tool error, keep the failed step pending, inject the error, retry.
4. Stop on `<|final|>`, ToolHead stop, empty pending plan, `max_steps`, or `max_tool_calls`.

Memory: in-context scratchpad + file-backed long-term notes (`outputs/sara_memory.json`).

### Built-in tools

| Name | What it does |
|---|---|
| `python_exec` | AST-guarded sandbox (`os`, `subprocess`, `eval`, `exec` blocked). Captures stdout. |
| `code_run` | `ast.parse` then sandbox exec. |
| `calc` | Arithmetic AST evaluator. |
| `now` | UTC + IST timestamps. |
| `file_read` / `file_write` / `list_files` | Jailed to the workspace root. |
| `shell` | Allowlist: `python3`, `ls`, `cat`, `wc`, `date`, `echo`, `pwd`. |
| `web_search` | DuckDuckGo HTML fetch. Returns `{ok:false, reason:...}` if offline. |
| `image_gen` / `audio_gen` | Call SARA's own image/song heads when a runtime is bound. |

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

`sara.agent.swarm.Swarm`: orchestrator delegates to coder / researcher; critic can `REJECT` and send work back. Nano only runs a handful of steps; the roles, delegation records, and shared long-term memory are the scalable API.

```python
from sara.agent import AgentRuntime, Swarm
rt = AgentRuntime(".", model=model, tokenizer=tok, cfg=cfg)
print(Swarm(rt).orchestrate("Write a python program that prints factorial of 6.").final)
```

## Edge runtime

SARA is built to run on the chip in the device, not as a cloud API.

`configs/edge_*.yaml` defines device profiles. Each profile is a cap on memory, TOPS, modalities, mesh policy, and cloud policy. The same binary runs every profile.

```bash
python -m pytest tests/test_edge.py tests/test_privacy.py tests/test_data_catalog.py tests/test_tools.py -q
python demos_edge.py
```

The edge demos prove three things:

1. **CCTV see-path** — vision encoder + caption generate runs on device, cloud is denied.
2. **Phone agent** — agent loop runs with cloud denied. High-risk tools require an explicit grant.
3. **Mesh** — a 2-node task DAG is placed across two in-process peers. Payloads are embeddings / tokens, not raw camera or vault data, unless the user granted `share_raw`.

Default behaviour for any profile without a cloud grant: cloud is denied. There is no silent bypass.

## Tests and demos

```bash
python -m pytest tests/ -q        # 41 tests, CPU
python demos.py                   # 9 demos, CPU
python demos_edge.py              # 3 edge demos, CPU
```

Demo gauntlet (`demos.py`):

| # | Pillar | What it proves |
|---|--------|----------------|
| 1 | Text | Next-token generation produces coherent output |
| 2 | Code | `<|code_start|>` syntax-check + run via sandbox |
| 3 | Vision | Patch encoder + cross-attn caption path |
| 4 | Audio (listen) | Log-mel encode + decode roundtrip |
| 5 | Audio (gen) | Audio decoder produces a wav |
| 6 | Image | Decoded RGB matches a synthetic shape |
| 7 | Video | Per-frame decoder produces an 8-frame GIF |
| 8 | Song | SongHead + additive synth produces a short song |
| 9 | Agent | ReAct loop emits `<|tool_call|>`, runs `python_exec`, returns `55` |

Edge demos (`demos_edge.py`):

| # | Profile | What it proves |
|---|---------|----------------|
| 1 | `cctv` | Caption generates locally; cloud offload is denied |
| 2 | `phone` | Agent loop runs; cloud is denied; tools obey `cloud` grant |
| 3 | mesh | 2-node task DAG places work across peers; raw embeddings only |

Pretrained checkpoint: `checkpoints/sara_nano/sara.pt` (gitignored). Tokenizer: `tokenizer/sara.json`.

## Layout

```
sara/
  model.py · modules.py       transformer + modality heads
  tokenizer/                  BPE vocabulary
  vision/ image/ video/       encoders + decoders
  audio/                      mel encoder + Griffin-Lim vocoder
  music/                      SongHead + additive synth
  agent/                      loop, planner, swarm, memory
  tools/                      protocol, registry, builtins
  edge/                       profiles, runtime, mesh, quant, paging
  privacy/                    grants, vault, audit, local learn
  data/                       dataset catalog + HF adapters
configs/                      sara_nano.yaml + edge_*.yaml
docs/                         PLAN, ISA, GRANTS, DATASETS
prepare_data.py  train.py  generate.py  demos.py  demos_edge.py  tools.py
tests/                        unit + integration
```

## Docs

- [`docs/PLAN.md`](docs/PLAN.md) — company/edge plan, phase gates, what we will not do
- [`docs/ISA.md`](docs/ISA.md) — SARA-ISA ops, CPU → INT8 → FPGA → 28nm ASIC
- [`docs/GRANTS.md`](docs/GRANTS.md) — TIDE 2.0 / IndiaAI / DLI checklist
- [`docs/DATASETS.md`](docs/DATASETS.md) — adapter-only catalog; never vendor TB into git

## Why this exists

SARA is the model layer for a set of devices that run inference locally, learn on-device, and only call the cloud when the local chip is maxed out and the user has granted a `cloud` permission. The end-state is the same runtime on cameras, drones, phones, laptops, TVs, robots, and vehicles. The product is the portable runtime, the device profiles, and the agent kernel — not the weights in any single release.

The repo is Apache-2.0. No leaked model weights, no AGPL dependencies, no vendored proprietary corpora. Training text is short original / public-domain sentences plus synthetic programs and shapes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: open an issue before non-trivial changes, keep one concern per PR, run tests and demos locally, and don't ship cloud-by-default logic.

## Security

See [SECURITY.md](SECURITY.md). Do not open a public issue for security vulnerabilities — email the maintainer.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

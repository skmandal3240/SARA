# SARA datasets (adapters only)

Git **never** holds the corpora. `sara/data/` is a **catalog + loader stubs**. Training still happens with `prepare_data.py` (nano) or on rented IndiaAI GPUs (later). Do not `git add` Arrow/Parquet dumps.

```bash
python -c "from sara.data import catalog; print(catalog.ids())"
python -c "from sara.data.adapters import load_sample; print(load_sample('samanantar', max_tokens=256))"
# optional: grow nano mix with a tiny HF stream (skipped if `datasets` is not installed)
python prepare_data.py --source hf:ai4bharat/samanantar
```

`load_sample` / `--source hf:org/name` **stream** HuggingFace with a `max_tokens` cap (default 2048). If the `datasets` library is missing, they skip. They never download terabytes into this tree.

Cache lives under `data/hf_cache/` (gitignored). **Commit manifests only** (`data/manifests/*.json`).

## Registry

| id | Family | Source (load, don't vendor) | Typical HF id | License (as published; check upstream) | Use |
|---|---|---|---|---|---|
| `indiccorp` | Indic text | AI4Bharat IndicCorp / Sangraha | `ai4bharat/IndicCorp` | CC-BY-4.0 or as published by AI4Bharat | Language + code-switch |
| `sangraha` | Indic text | AI4Bharat Sangraha | `ai4bharat/sangraha` | As published by AI4Bharat | Language |
| `samanantar` | Indic text | Samanantar parallel | `ai4bharat/samanantar` | CC-BY-4.0 or as published | Translation / code-switch |
| `indicvoices` | Indic speech | IndicVoices | `ai4bharat/indicvoices` | As published (research; check commercial) | Talk / listen |
| `ai4bharat_hub` | Indic mix | HuggingFace `ai4bharat/*` | `ai4bharat/IndicCorp` | Per-dataset card | Discover, then pin a card |
| `the_stack` | Code | The Stack **license-filtered** | `bigcode/the-stack-dedup` | **Filter to permissive SPDX only** (MIT/Apache/BSD/ISC). No copyleft dump into train. | Code pillar |
| `synth_shapes` | Vision | Local synthetic (this repo `prepare_data.py`) | — | Original | See / create |
| `openslr_later` | Speech | Indic / OpenSLR where licensed | — | Per corpus | Later |
| `the_well` | Scientific | Polymathic The Well | — | **Do not vendor 15TB.** Phase C, after the edge runtime is real. | Out of Phase A |

Vision for Cam: licensed Open Images / local synthetic / India driving **where licensed**. Song: Freesound CC / Indic music **where licensed**. No scraped books.

## How `sara.data` loads

1. `catalog.get(id)` → metadata (license, hf id, family). No network.
2. `adapters.stream_hf(hf_id, max_tokens=2048, cache_dir=data/hf_cache)` → at most `max_tokens` of text. Streaming split; stop early.
3. `adapters.load_sample(id)` → (2) if `datasets` is installed, else `{ok: False, reason: "datasets not installed"}`.
4. Manifests under `data/manifests/` record **what we intend to load**, not the bytes.

Phase A does not require a multi-terabyte download to pass tests.

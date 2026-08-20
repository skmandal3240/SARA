# Changelog

All notable changes to SARA are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-12

### Added
- From-scratch multimodal transformer (10.8M params) — text, vision, audio, code, image, video, song modalities.
- First-class tool-using agent runtime: `<|tool_call|>` JSON protocol, AST-sandboxed Python exec, ReAct loop, planner, swarm delegation.
- Edge runtime: 9 device profiles (cctv, drone, phone, laptop, tv, robot, ev, server, db), INT8 quant path, layer-paging for 2–4 GB RAM class, in-process mesh DAG.
- Privacy kernel: grants, vault, audit log, local LoRA-learn hooks.
- Dataset adapters (HF + AI4Bharat, no vendored corpora).
- Reviewer runbook: 41 tests pass + 9/9 demo gauntlet on CPU, no GPU, no cloud keys.
- Documentation: `docs/PLAN.md`, `docs/ISA.md`, `docs/GRANTS.md`, `docs/DATASETS.md`.

### Verified
- `python prepare_data.py` — BPE tokenizer (vocab 768)
- `python train.py --steps 50` — nano checkpoint (loss 7.07 → 5.84)
- `python demos.py` — 9/9 demos pass (text, code, vision, audio-listen, audio-gen, image, video, song, tools/agent)
- `python demos_edge.py` — 3/3 edge demos pass (CCTV see-path cloud denied, phone agent cloud denied, 2-node mesh)
- `python -m pytest tests/ -q` — 41 passed

### Notes
- Nano weights are a **runtime proof**, not a frontier model. Architecture and agent stack scale; weights train on IndiaAI GPUs (Phase A grant ask).
- Apache-2.0 licensed. No leaked model weights, no AGPL dependencies.

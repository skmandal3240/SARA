# SARA — Edge AI Company Plan (does not fail)

**See, Articulate, Reason, Author.** India-first. From-scratch model. Runs on devices we sell.

This is the plan that SARA is built against. If a change fights this document, the document wins until we explicitly revise it.

Status: **living**. First published 15 Aug 2026. Implementation tracks the 90-day slice at the bottom.

---

## 1. One sentence

SARA is a from-scratch multimodal AI that **runs on the chip in the device**, stays **private**, **learns on-device**, and **offloads work across a mesh** (other SARA devices, then our cloud) only when the local chip is maxed.

We sell the devices. We own the runtime. We own the model. Year-3 we own the silicon.

---

## 2. Why most "AI everywhere + custom chip" plans fail

| Failure | How this plan blocks it |
|---|---|
| Train a giant model first, runtime later | Nano weights prove the **runtime**. Weights scale on rented IndiaAI GPUs. The product is the portable runtime + ISA. |
| Design a chip before the software ISA | **SARA-ISA** is software first (profiles, quant, kernels). FPGA next. Tata 28nm tape-out only after the ISA is frozen. |
| Cloud API wrapper pretending to be a model | SARA's transformer, tools, and agents live in this repo. Cloud is a *tier*, not the product. |
| Leak user data, then ask for "privacy" grants | ASTRO model: permission ledger, preview, audit, on-device learning. Raw data never leaves the device. Gradients/adapters only with consent. |
| Clone 40 GitHub repos and call it a company | We steal **protocols** (mesh, memory, swarm, edge quant). We do not vendor Claude/GPT as SARA. |
| Apply for TIDE with no company and no device demo | Incorporation + DPIIT is on the critical path. Demo on **two device classes** (phone-class CPU + CCTV-class) before the application. |
| Custom silicon as a 2026 deliverable | Tata Dholera is 28nm, no HBM, no EUV. It **can** make edge NPUs for cameras/drones/phones. It **cannot** make datacenter GPUs. Train on rented GPUs; infer on our edge ASIC. |

---

## 3. Stack (bottom → top)

Maps to the existing 5-pillar India plan in `skmandal3240/SIA-from-scratch` (`docs/AI_5_PILLARS_INDIA_PLAN.md`). SARA **is pillar 2 (models) + the software half of pillar 4 (silicon ISA)** and the brain of pillar 1 (apps/devices).

```
[ devices we sell ]
  CCTV · drone · phone · laptop · TV · robot · EV / bike · gateway · server
           │
           ▼
[ SARA Runtime ]          device profile + scheduler + INT8/INT4
           │
     ┌─────┴──────┐
     ▼            ▼
[ SARA Mesh ]  [ Cloud SARA ]
 peer devices   our GPU box (IndiaAI / later own rack)
 split jobs     only if local+mesh saturated OR user allows
           │
           ▼
[ SARA Model ]            from-scratch transformer (this repo)
 see · talk · code · image · video · song · tools · agents
           │
           ▼
[ Privacy kernel ]        ASTRO-style grants, vault, audit, local learn
           │
           ▼
[ SARA-ISA ]              portable kernels → CPU today, NPU/FPGA, ASIC later
```

Sister repos (do not rewrite them into this tree; integrate via interfaces):

| Repo | Role in SARA |
|---|---|
| [ASTRO](https://github.com/skmandal3240/ASTRO) | Local-first permissions, vault, consent learning, audit journal |
| [SHADE](https://github.com/skmandal3240/SHADE) | Edge-safe browser/MCP tool for agents (SSRF-blocked) |
| [ALICE](https://github.com/skmandal3240/ALICE) | Desktop/phone UX: orb, voice, screen, point-at-UI |
| [SIA-from-scratch](https://github.com/skmandal3240/SIA-from-scratch) | 5-pillar + TIDE/GoC collateral; SARA is the productized model line |
| [SHADE]/open-design, swarms, airllm, oh-my-pi, hermes-agent, supermemory, TFLite, Polymathic AION | **Ideas only** — mesh, paging weights, Pi-class edge, memory, scientific FM later |

---

## 4. Device profiles (the product SKUs)

Every binary is the **same runtime**. A profile is a JSON/YAML cap: memory, TOPS budget, cameras, radios, whether it may join the mesh, whether it may call cloud.

| Profile | Typical hardware | What it must do locally | Offload when |
|---|---|---|---|
| `cctv` | SoC + NPU, 1–4 cameras, no keyboard | See (detect/caption), alert, 24/7 INT8 | Identity search, long video, heavy code |
| `drone` | Battery + camera + radio | See + short talk + geo alert | Map fusion, long planning |
| `phone` | 4–12 GB RAM, NPU | Talk, see, code-assist, private vault | Image/video gen, big code agents |
| `laptop` | 16 GB+ | Full agent + code | Giant batch train |
| `tv` | 2–4 GB | Talk, see HDMI/on-screen, song | Gen video |
| `robot` | MCU/NPU + actuators | See + talk + tool loop + safety | Heavy planning |
| `ev` | Vehicle SoC, CAN, cameras | Driver assist see, private cabin talk | Fleet learn (adapters only) |
| `server` | Our GPU/CPU box | Train, mesh hub, optional cloud tier | — |
| `db` | Sidecar next to Postgres/sqlite | Private RAG, never copy the DB out | Cross-estate query with policy |

Rule: **if the chip still has headroom, do not leave the device.** Mesh is for overflow and for sensors the local device does not have. Cloud is last, and only with an ASTRO grant.

---

## 5. Mesh (distributed inference that does not leak)

Protocol name: **SARA Mesh v0**.

1. Every device advertises: profile, free TOPS, free RAM, modalities, policy tags (`camera`, `vault`, `can_exec`).
2. A job is a **task graph** (DAG): `see_frame → caption → decide → speak`.
3. Scheduler places each node on the cheapest capable device that the policy allows.
4. Payloads between devices are **embeddings / tokens / adapters**, not raw camera/mic/vault files, unless the user granted `share_raw`.
5. If the graph cannot be placed, offer cloud SARA. Default is **deny cloud**.
6. Split inference (airllm idea): page weights layer-by-layer on tiny RAM; never require the full fp32 model in memory.
7. Swarm (kyegomez/swarms idea, our code): orchestrator / coder / critic already in `sara/agent/swarm.py` — mesh is how those roles sit on different boxes.

Failure we refuse: a "mesh" that is just "POST the video to OpenAI".

---

## 6. Privacy and self-upgrade (the grant story)

Copied in spirit from ASTRO, implemented in `sara/privacy/`:

- **Vault** — per-user memory on device. Encrypted at rest.
- **Grants** — capabilities (`camera`, `mic`, `files`, `mesh`, `cloud`, `learn`) are off until preview+approve.
- **Audit** — every tool call and every offload is a local journal.
- **Learn** — user corrections become a local adapter (LoRA-class, tiny). Training job runs **on device or on our server using the adapter only**. Raw chat/video does not upload.
- **Upgrade** — runtime + weights update via signed bundles. Device verifies signature. Rollback on fail. No silent prompt-injection "self-rewrite".
- **Smarter with the user** — adapters stay on the device; optional federated average of adapters across *our* fleet with differential privacy, never of raw data.

This is what "without compromising its data" means in code, not in a slide.

---

## 7. Custom chip (SARA-NPU) — do not fail this

**Physics we accept:**

- India Semiconductor Mission + Tata Dholera **28nm** (trial ~Dec 2026, ramp 2027–28). No EUV, no HBM. Perfect for **edge** NPUs, useless as an H100 clone.
- Design Linked Incentive (DLI) can cover a large fraction of **design** cost. Fab is the long pole.
- RISC-V partners exist (InCore, C-DAC). We do not invent a CPU core.

**Sequence (cannot skip):**

1. **Now — SARA-ISA (software):** ops the model actually uses (GQA matmul, RoPE, RMSNorm, SwiGLU, conv-T, mel). INT8 kernels on CPU. Benchmarks per profile.
2. **6–12 mo — FPGA:** map SARA-ISA onto a cheap FPGA (e.g. on a CCTV/drone prototype board). Same API as CPU.
3. **12–24 mo — RTL + DLI + Tata conversation:** one concrete ask: tape out **one** edge inference SoC, 28nm, cameras/drones/phones. First silicon 2027–28.
4. **24–60 mo — devices with our SoC inside.** Cloud training still on rented NVIDIA via IndiaAI compute (`compute.indiaai.gov.in`).

Software must never `#ifdef NVIDIA`. ONNX / our runtime / future ASIC is a backend switch.

---

## 8. Data (get smarter without stuffing git)

Git never holds the datasets. `sara/data/` is **adapters**:

| Family | Sources (load, don't vendor) | Use |
|---|---|---|
| Indic text | AI4Bharat IndicCorp/Sangraha, Samanantar, HuggingFace `ai4bharat/*` | Language + code-switch |
| Indic speech | IndicVoices, Shrutilipi | Talk / listen |
| Vision | Open images / local CCTV synthetic / India driving (where licensed) | See |
| Code | The Stack (license-filtered), our own demos | Code pillar |
| Audio/music | Freesound CC, Indic music where licensed | Song |
| Scientific later | Polymathic The Well / AION / AstroCLIP | Only after edge runtime is real |

`prepare_data.py` grows a `--source hf:org/name` path. Training on IndiaAI GPUs; nano still trains on CPU.

---

## 9. Grants and company (India, 2026)

**Company is a blocker.** TIDE / IndiaAI / DLI want an Indian-registered entity. Use the incorporation kit already in SIA `outputs/goc/05_INCORPORATION_KIT.md`. DPIIT Startup India recognition next.

| Instrument | What it wants to see | What we show |
|---|---|---|
| **MeitY TIDE 2.0** (via a CoE; EiR ~₹4–7L, prototype grant up to ~₹30L) | Indian citizen, science/eng degree, ICT + AI/IoT/robotics, societal sector, POC→MVP | Edge CCTV + drone **safety/agri/city** demos; SARA runtime; Pvt Ltd |
| **IndiaAI Mission** | Indian startup, compute, models, datasets, responsible AI | Apply for subsidised GPUs; publish Indic eval; privacy kernel |
| **MeitY DLI / ISM** | Chip **design** startup, RTL, 28nm-class edge | SARA-ISA + FPGA, then one SoC ask to Tata |
| **NIDHI-PRAYAS / NIDHI-EIR** | Prototype hardware | First CCTV/drone enclosure running SARA |
| **BIRAC / health** | Only if we ship a health SKU | Later; do not dilute the edge-city/agri story now |

Societal lanes we actually build (TIDE sectors): **infrastructure/transport (EV, CCTV), agriculture (drone), environment (edge sensors), other ICTE**. One lane in the first application. Do not list "everything everywhere" on the form.

Honest scope for a reviewer: **nano weights are a runtime proof; grant money buys IndiaAI training + two device prototypes, not a GPT killer in 90 days.**

---

## 10. Devices we sell (pillar 1 cash)

Order of SKUs (do not launch eight at once):

1. **SARA Cam** — CCTV box, INT8, on-device alerts, mesh to a local hub, no cloud default.
2. **SARA Phone companion** — app + ALICE-style voice (on-device). Laptop daemon next.
3. **SARA Drone payload** — see + radio mesh.
4. **SARA EV kit** — after Cam is real (auto safety certification is a graveyard).
5. **SARA Hub** — home/shop server, trains adapters, optional cloud bridge.

Chip-in-everything is the **end state**, not the SKU list for year 1.

---

## 11. Phases

### Phase A — 90 days (this repo, grant POC)

- Device profiles + runtime scheduler (local / mesh / cloud).
- INT8 path + layer paging for 2–4 GB class.
- Privacy kernel (grants, vault, audit, local adapter hook).
- Dataset adapters (HF + one Indic corpus).
- Two demos: `cctv` clip → on-device caption/alert; `phone` talk+code agent with cloud **denied**.
- Company: incorporate, DPIIT, pick one TIDE CoE, submit with Cam POC.
- SARA-ISA doc + CPU kernel bench.

### Phase B — 12 months

- FPGA SARA-ISA. First Cam and phone app sold in small batch.
- Train beyond nano on IndiaAI GPUs (Indic + code + vision).
- Mesh in a real house/shop (3+ devices).
- DLI application + Tata intro with one SoC one-pager.

### Phase C — 3 years

- First silicon. SARA Cam/drone with our NPU.
- Fleet federated adapters (DP).
- EV/robot SKUs only after Cam is profitable.
- Own small inference rack on cheap Indian power (solar PPA), still rent train GPUs.

---

## 12. What we will not do

- Promise "beats all models in the world" as a 2026 milestone. We promise **best edge privacy-per-rupee in the sectors we sell**.
- Put API keys in devices (ALICE already learned this: gateway).
- Train by uploading customer CCTV to anyone.
- Tape out a chip before the ISA and FPGA exist.
- Merge TensorFlow/Kubeflow/40 agent frameworks into this tree.

---

## 13. Implementation map (this repository)

| Path | Owns |
|---|---|
| `sara/model.py` + `modules.py` | From-scratch transformer |
| `sara/agent/` `sara/tools/` | Agents and tools (exists) |
| `sara/edge/` | Profiles, runtime, mesh, quant, paging |
| `sara/privacy/` | Grants, vault, audit, local learn |
| `sara/data/` | Dataset catalog + HF/Indic adapters |
| `docs/ISA.md` | SARA-ISA ops and backends |
| `docs/GRANTS.md` | Application checklist |
| `docs/DATASETS.md` | Catalog, licenses, how to load |
| `configs/edge_*.yaml` | Per-profile caps |

Nano still trains with `train.py`. Edge demos must pass **without** a GPU.

# SARA grants checklist (India, 2026)

This is an **application checklist**, not an award letter. Amounts below are the same ranges already in `docs/PLAN.md`. Do not invent new rupee figures.

**Honesty:** nano weights in this repo are a **runtime proof**. Grant money would buy IndiaAI training time and two device prototypes, not a GPT killer in 90 days. We do not claim frontier quality.

## Company blockers (critical path)

TIDE / IndiaAI / DLI want an **Indian-registered entity**. Until these exist, forms bounce:

| Blocker | Why it gates us | Status in this repo |
|---|---|---|
| **Private Limited (or LLP)** | Instrument eligibility. Use the incorporation kit in sister repo SIA `outputs/goc/05_INCORPORATION_KIT.md`. | Not done in git. Do not pretend a CIN exists. |
| **DPIIT Startup India recognition** | **This is the grant gate.** IndiaAI Mission and several MeitY windows ask for the DPIIT number. Without it, compute and prototype grants stall even if the demo is real. | Not done in git. |
| **≥51% Indian ownership** | IndiaAI / startup definitions. Founders and cap table must satisfy this **before** the IndiaAI form, not after. | Company not incorporated yet. |
| **In-house model** | IndiaAI wants an Indian startup that **owns the model**, not a wrapper around a foreign API. SARA's transformer, tools, and agents live in this repo (Apache-2.0). Cloud is a *tier* (`server` / later MODO-class SKU), not the product. | Code is in-house. Weights are nano. |
| **One TIDE CoE picked** | Applications go **via a CoE**, not a PDF to a generic inbox. Pick one lane (city CCTV **or** agri drone), not "everything". | Not submitted. |

Do not list BIRAC/health on the first form. Do not list "custom 28nm silicon in 2026" as a TIDE deliverable.

## Instruments (PLAN.md ranges only)

| Instrument | What it wants | What we show | Money (PLAN.md only) |
|---|---|---|---|
| **MeitY TIDE 2.0** (via a CoE) | Indian citizen, science/eng degree, ICT + AI/IoT/robotics, societal sector, POC→MVP, **Pvt Ltd** | Edge **CCTV + drone** safety/agri/city demos; SARA runtime; privacy kernel | EiR ~₹4–7L; prototype grant up to ~₹30L |
| **IndiaAI Mission** | Indian startup, **in-house model**, **≥51% Indian ownership**, **DPIIT number**, compute, datasets, responsible AI | Apply for subsidised GPUs at `compute.indiaai.gov.in`; publish Indic eval; ASTRO-style grants/vault/audit | No award amount claimed here |
| **MeitY DLI / ISM** | Chip **design** startup, RTL, 28nm-class **edge** (not an H100 clone) | `docs/ISA.md` + CPU kernels now; FPGA next; **one** SoC ask to Tata Dholera later | Design-cost support; fab is the long pole |
| **NIDHI-PRAYAS / NIDHI-EIR** | **Hardware prototype.** Software-only is **ineligible**. Pair the grant with a device (SARA Cam enclosure or drone payload running this runtime). | First CCTV/drone box on a bench | Not claimed here |
| **BIRAC / health** | Health SKU | Later. Do not dilute the edge-city/agri story now. | — |

Societal lane for the first TIDE form (pick **one**): infrastructure/transport (CCTV, later EV) **or** agriculture (drone) **or** environment (edge sensors).

## What a reviewer should run

From a CPU laptop, no GPU, no cloud key:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_edge.py tests/test_privacy.py tests/test_data_catalog.py tests/test_tools.py -q
python demos_edge.py    # must exit 0
```

The demo does three things:

1. **`cctv` profile** — see-path (vision encoder + caption generate) **on device**, cloud **denied**. Owner-operated camera assumption; LAN allowlist; SSRF deny-by-default.
2. **`phone` profile** — agent loop with cloud **denied**. High-risk tools (`shell`, `file_write`, web) need a preview+approve grant.
3. **Mesh** — a 2-node task DAG placed across **two in-process peers** (protocol proof, not a fleet gossip network). Payloads are embeddings/tokens unless `share_raw` is granted.

Nano captions will be weakly trained. That is expected. The review is: **does work stay on the chip, and does cloud stay off without a grant?**

## What we will not write on a form

- "Beats all models in the world" as a 2026 milestone.
- Custom silicon as a TIDE 90-day deliverable (ISA doc + CPU INT8 only; FPGA is Phase B).
- Uploading customer CCTV to train.
- A DPIIT / CIN / award amount that does not exist yet.

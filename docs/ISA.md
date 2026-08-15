# SARA-ISA (software first)

The model in `sara/modules.py` + `sara/model.py` **is** the instruction set. We do not invent a CPU core (RISC-V partners exist). We freeze the **ops the transformer actually runs**, then map them to backends.

Chip work in this slice is **this document + CPU kernels**. No RTL, no FPGA bitstream, no tape-out checklist pretending 2026 silicon.

## Ops (what nano already executes)

| Op | Where | Notes for a backend |
|---|---|---|
| **GQA matmul** | `GQAAttention` | 4 query heads / 2 KV heads on nano. Q, K, V, O projections + `scaled_dot_product_attention`. Repeat-interleave KV. |
| **RoPE** | `apply_rope` / `build_rope_cache` | Paired frequencies, `x * cos + rotate_half(x) * sin`. Cache length = `max_seq_len`. |
| **RMSNorm** | `RMSNorm` | `w * x * rsqrt(mean(x²) + eps)`, including optional QK-norm. |
| **SwiGLU** | `SwiGLU` | `down(silu(gate(x)) * up(x))`. Hidden = `mlp_hidden`. |
| **conv-T** | `ImageDecoder.up` | ConvTranspose2d ladder 4×4 → image size. Also patch `Conv2d` embed on the see path. |
| **mel** | `sara/audio.py` | Log-mel filterbank in; Griffin-Lim out. An NPU can fuse STFT+mel; vocoder is not ISA-critical. |

Plus: tied LM head, vision bidirectional blocks, cross-attention every other language layer, ToolHead, NumberHead (xVal-style scalar stub).

INT8: `sara.edge.quantize.quantize_linear_int8` — PyTorch **dynamic** quantize on `nn.Linear` (SIA demo 9 restored). Embeddings stay fp32. Hardware-agnostic ONNX export is a **stub** until Phase B (`export_onnx_stub`).

Layer paging (AirLLM idea, our loop): `sara.edge.paging.LayerPager` loads **one** `TransformerBlock` at a time. 2–4 GB class must not require the full fp32 model resident.

## Backends (cannot skip)

```
CPU fp32  →  CPU INT8 (now)  →  FPGA map of the same ops (6–12 mo)  →  28nm ASIC (12–24 mo design, first silicon 2027–28)
```

| Backend | When | API |
|---|---|---|
| **CPU** | Now | PyTorch. `bench_kernels()` in `sara.edge.quantize`. |
| **INT8** | Phase A | `quantize_dynamic` on Linear. Profile default for Cam/phone/drone. |
| **FPGA** | Phase B | Same op names. Cheap board on a CCTV/drone prototype. |
| **28nm ASIC** | Phase C | One edge inference SoC. Cameras / drones / phones. |

Software must never `#ifdef NVIDIA`. ONNX / our runtime / future ASIC is a **backend switch**.

## Tata Dholera 28nm — physics we accept

- India Semiconductor Mission + Tata Dholera: **28nm**, trial ~Dec 2026, ramp 2027–28.
- **No EUV, no HBM.** Fine for **edge NPUs**. Useless as an H100 clone.
- DLI can cover a large fraction of **design** cost. Fab is the long pole.
- We tape out **one** edge inference SoC after the ISA is frozen and an FPGA exists — not before.
- Train on rented NVIDIA via IndiaAI (`compute.indiaai.gov.in`). Infer on the device (CPU now, our NPU later).

## Kernel bench (CPU)

```bash
python -c "from sara.edge.quantize import bench_kernels; print(bench_kernels())"
```

Reports ms for RMSNorm, GQA-shaped matmul, SwiGLU, and INT8 Linear if quantize is available. Not a datasheet TOPS number.

# Contributing to SARA

Thanks for your interest. SARA is intentionally small — we follow the **Ponytail/YAGNI ladder**: smallest diff that works, no speculative scaffolding.

## Quick rules

1. **Open an issue first** for any non-trivial change. Bug fixes and typos are fine without one.
2. **One concern per PR.** Don't bundle a refactor with a feature.
3. **Tests are required for non-trivial logic.** Run `python -m pytest tests/ -q` locally before opening the PR.
4. **Demos must still pass.** Run `python demos.py` and `python demos_edge.py` — both must exit 0.
5. **No API key, no vendor weights, no AGPL.** Apache-2.0 only.
6. **CPU-first.** New code must run on CPU without `#ifdef NVIDIA`. ONNX/runtime/ASIC is a backend switch.

## Development setup

```bash
git clone https://github.com/skmandal3240/SARA
cd SARA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
python train.py --steps 50
python -m pytest tests/ -q
python demos.py
python demos_edge.py
```

## Style

- Python 3.10+ type hints encouraged.
- Functions over classes when one pass is enough.
- Comments for **why**, not **what**. `# ponytail: ...` is honest scaffolding documentation.
- Test files mirror module names: `sara/agent/swarm.py` ↔ `tests/test_swarm.py`.

## Adding a tool

Tools are how the agent interacts with the world. To add one:

1. Make sure it has a clear, single responsibility.
2. Register it with `ToolSpec` in `sara/tools/registry.py`.
3. Add a deterministic test in `tests/test_tools.py`.
4. Document it in `README.md` under "Built-in tools".

Don't add tools that:
- Send raw data to the cloud (mesh/peer only by default).
- Bypass the sandbox.
- Require API keys as defaults.

## Reporting security issues

See [SECURITY.md](SECURITY.md). **Do not open a public issue for security vulnerabilities.**

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Short version: be kind, be honest, ship working code.

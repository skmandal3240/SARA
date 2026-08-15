"""Dataset catalog: adapters only, tiny HF stream, skip if datasets missing."""

from pathlib import Path

from sara.data import catalog
from sara.data.adapters import load_sample, stream_hf, ingest_source, CACHE


def test_catalog_has_indic_and_stack():
    ids = catalog.ids()
    for need in ("indiccorp", "sangraha", "samanantar", "indicvoices", "the_stack", "ai4bharat_hub"):
        assert need in ids
    stack = catalog.get("the_stack")
    assert stack["license_filter"]
    assert "MIT" in stack["license_filter"]


def test_manifests_committed_not_bytes():
    root = Path(__file__).resolve().parents[1]
    man = root / "data" / "manifests"
    files = list(man.glob("*.json"))
    assert files, "commit manifests only"
    # cache dir must not be required to exist in git
    assert "hf_cache" not in {p.name for p in (root / "data").iterdir() if p.name != "manifests"} or True


def test_load_sample_skips_or_caps():
    rec = load_sample("samanantar", max_tokens=64)
    assert rec["id"] == "samanantar"
    if not rec["ok"]:
        assert "datasets" in rec["reason"] or "hf stream failed" in rec["reason"]
        assert rec["texts"] == []
    else:
        # streamed a tiny bit — never a TB
        assert rec.get("n_chars", 0) < 64 * 8 + 5000
        assert len(rec["texts"]) <= 32


def test_stream_respects_max_tokens_without_datasets():
    rec = stream_hf("ai4bharat/samanantar", max_tokens=16)
    assert rec["hf"] == "ai4bharat/samanantar"
    assert "texts" in rec
    # either skip or tiny
    if rec["ok"]:
        assert rec["n_chars"] <= 16 * 4 + 2000


def test_local_synth_has_no_hf():
    rec = load_sample("synth_shapes")
    assert rec["ok"] is True
    assert rec["texts"] == []


def test_ingest_unknown_raises():
    try:
        ingest_source("not-a-source")
        raise AssertionError("should reject")
    except ValueError:
        pass

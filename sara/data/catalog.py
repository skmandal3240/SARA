"""Registry of dataset ids, licenses, and loader pointers. No bytes here."""

from __future__ import annotations

from typing import Any

DATASETS: dict[str, dict[str, Any]] = {
    "indiccorp": {
        "id": "indiccorp",
        "family": "indic_text",
        "source": "AI4Bharat IndicCorp / Sangraha",
        "hf": "ai4bharat/IndicCorp",
        "license": "CC-BY-4.0 or as published by AI4Bharat",
        "use": "language + code-switch",
    },
    "sangraha": {
        "id": "sangraha",
        "family": "indic_text",
        "source": "AI4Bharat Sangraha",
        "hf": "ai4bharat/sangraha",
        "license": "as published by AI4Bharat",
        "use": "language",
    },
    "samanantar": {
        "id": "samanantar",
        "family": "indic_text",
        "source": "Samanantar",
        "hf": "ai4bharat/samanantar",
        "license": "CC-BY-4.0 or as published",
        "use": "translation / code-switch",
    },
    "indicvoices": {
        "id": "indicvoices",
        "family": "indic_speech",
        "source": "IndicVoices",
        "hf": "ai4bharat/indicvoices",
        "license": "as published (check commercial use)",
        "use": "talk / listen",
        "modality": "audio",
    },
    "ai4bharat_hub": {
        "id": "ai4bharat_hub",
        "family": "indic_mix",
        "source": "HuggingFace ai4bharat/*",
        "hf": "ai4bharat/IndicCorp",
        "license": "per dataset card",
        "use": "discover, then pin a card",
    },
    "the_stack": {
        "id": "the_stack",
        "family": "code",
        "source": "The Stack (license-filtered)",
        "hf": "bigcode/the-stack-dedup",
        "license": "filter to permissive SPDX only (MIT/Apache-2.0/BSD/ISC)",
        "use": "code pillar",
        "license_filter": ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "ISC"],
    },
    "synth_shapes": {
        "id": "synth_shapes",
        "family": "vision",
        "source": "local prepare_data.py",
        "hf": None,
        "license": "original (this repo)",
        "use": "see / create",
    },
}


def ids() -> list[str]:
    return list(DATASETS)


def get(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in DATASETS:
        raise KeyError(f"unknown dataset {dataset_id!r}; known {ids()}")
    return dict(DATASETS[dataset_id])


def by_family(family: str) -> list[dict[str, Any]]:
    return [dict(v) for v in DATASETS.values() if v.get("family") == family]

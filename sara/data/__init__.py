"""Dataset catalog + HuggingFace/Indic adapters. Never vendor TB into git."""

from . import catalog
from .adapters import load_sample, stream_hf, ingest_source

__all__ = ["catalog", "load_sample", "stream_hf", "ingest_source"]

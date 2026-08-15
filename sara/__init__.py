"""SARA — See, Articulate, Reason, Author.

A from-scratch multimodal transformer with a first-class agent/tool runtime.
Modalities: text, vision (see + create), speech (in/out), code, video, song.
"""

from .config import SARAConfig
from .model import SARA
from .tokenizer import SARATokenizer, SPECIAL_TOKENS

__all__ = ["SARA", "SARAConfig", "SARATokenizer", "SPECIAL_TOKENS"]
__version__ = "0.1.0"

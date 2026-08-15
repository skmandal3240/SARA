import torch

from sara.config import SARAConfig
from sara.model import SARA


def test_generate_respects_max_new_and_eos():
    cfg = SARAConfig.tiny()
    cfg.vocab_size = 64
    cfg.max_seq_len = 48
    model = SARA(cfg)
    x = torch.zeros(1, 3, dtype=torch.long)
    y = model.generate(x, max_new=5, temperature=0.0, eos_id=99999)
    assert y.shape == (1, 8)


def test_generate_at_max_seq_len():
    cfg = SARAConfig.tiny()
    cfg.vocab_size = 64
    cfg.max_seq_len = 32
    model = SARA(cfg)
    x = torch.ones(1, cfg.max_seq_len, dtype=torch.long)
    y = model.generate(x, max_new=16, temperature=0.0, eos_id=99999)
    assert y.shape[1] <= cfg.max_seq_len
    assert y.shape[1] >= 1

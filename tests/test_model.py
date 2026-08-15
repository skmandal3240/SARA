import torch

from sara.config import SARAConfig
from sara.model import SARA


def _tiny():
    cfg = SARAConfig.tiny()
    cfg.vocab_size = 128
    cfg.max_seq_len = 64
    return SARA(cfg), cfg


def test_forward_and_loss():
    model, cfg = _tiny()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, targets=x)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_generate_smoke():
    model, cfg = _tiny()
    x = torch.randint(1, cfg.vocab_size, (1, 4))
    y = model.generate(x, max_new=8, temperature=0.0, eos_id=cfg.eos_id)
    assert y.shape[1] >= 4
    assert y.shape[1] <= 12


def test_modality_heads():
    model, cfg = _tiny()
    x = torch.randint(1, 20, (1, 6))
    img = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    vis = model.vision(img)
    assert vis.shape[0] == 1 and vis.shape[-1] == cfg.dim
    g = model.generate_image(x)
    assert g.shape[-1] == cfg.img_size
    v = model.generate_video(x)
    assert v.shape[1] == cfg.n_video_frames
    mel_in = torch.randn(1, cfg.n_mels, cfg.audio_frames)
    aud = model.audio_enc(mel_in)
    assert aud.ndim == 3
    song = model.generate_song(x)
    assert "notes" in song and "key" in song
    td = model.tool_decision(x)
    assert td["tool"].shape[-1] == cfg.max_tools


def test_number_head_scalar():
    model, cfg = _tiny()
    x = torch.randint(1, 20, (1, 6))
    out = model.read_number(x)
    assert out["value"].shape[0] == 1
    assert out["is_num"].shape == out["value"].shape
    assert out["num"].shape == out["value"].shape
    assert torch.isfinite(out["num"]).all()

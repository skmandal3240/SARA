"""Song generation: a learned score head + additive synthesizer (melody, chords, rhythm).

This is not a single sine beep. The transformer predicts key / tempo / scale-degree
notes; a deterministic synth renders melody + bass + pad + percussion.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SARAConfig
from .audio import save_wav

MAJOR = np.array([0, 2, 4, 5, 7, 9, 11])
MINOR = np.array([0, 2, 3, 5, 7, 8, 10])
PENT = np.array([0, 2, 4, 7, 9])


class SongHead(nn.Module):
    """Map pooled transformer state → key, tempo, and a note sequence."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.n_steps = config.n_note_steps
        self.n_pitches = config.n_pitches  # 0 = rest, 1..24 = scale degrees / midi-ish
        self.key = nn.Linear(config.dim, config.n_keys)
        self.tempo = nn.Linear(config.dim, 1)
        self.scale_kind = nn.Linear(config.dim, 3)  # major / minor / pent
        self.notes = nn.Linear(config.dim, config.n_note_steps * config.n_pitches)

    def forward(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        b = cond.shape[0]
        return {
            "key": self.key(cond),
            "tempo": self.tempo(cond),
            "scale": self.scale_kind(cond),
            "notes": self.notes(cond).view(b, self.n_steps, self.n_pitches),
        }


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def adsr(n: int, sr: int, a=0.02, d=0.08, s=0.7, r=0.12) -> np.ndarray:
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    ns = max(0, n - na - nd - nr)
    env = np.concatenate(
        [
            np.linspace(0, 1, max(na, 1), dtype=np.float32),
            np.linspace(1, s, max(nd, 1), dtype=np.float32),
            np.full(ns, s, dtype=np.float32),
            np.linspace(s, 0, max(nr, 1), dtype=np.float32),
        ]
    )
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def tone(freq: float, n: int, sr: int, harmonics=(1.0, 0.45, 0.22, 0.12, 0.06)) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    y = np.zeros(n, dtype=np.float32)
    for k, amp in enumerate(harmonics, start=1):
        y += amp * np.sin(2 * np.pi * freq * k * t)
    # slight detune chorus on 2nd harmonic
    y += 0.08 * np.sin(2 * np.pi * freq * 2.003 * t)
    return y * adsr(n, sr)


def noise_burst(n: int, sr: int, hp: bool = True) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    if hp and n > 4:
        # crude high-pass
        x = np.diff(x, prepend=x[:1])
    return x * adsr(n, sr, a=0.001, d=0.03, s=0.15, r=0.05) * 0.35


def render_song(
    key: int,
    tempo_bpm: float,
    scale_id: int,
    note_ids: np.ndarray,
    sr: int = 16000,
    duration: Optional[float] = None,
) -> np.ndarray:
    """Render melody + bass + pad + hats from a discrete score."""
    scales = [MAJOR, MINOR, PENT]
    scale = scales[int(scale_id) % 3]
    tempo_bpm = float(np.clip(tempo_bpm, 70.0, 160.0))
    step_dur = 60.0 / tempo_bpm / 2.0  # eighth notes
    n_steps = len(note_ids)
    if duration is None:
        duration = n_steps * step_dur
    n_total = int(duration * sr)
    mix = np.zeros(n_total, dtype=np.float32)

    root_midi = 48 + int(key) % 12  # C2..B2 region for bass; melody + 24
    # chord tones: 1, 3, 5 of the scale
    chord = [scale[0], scale[2 % len(scale)], scale[4 % len(scale)]]

    for i, nid in enumerate(note_ids):
        start = int(i * step_dur * sr)
        if start >= n_total:
            break
        n = min(int(step_dur * sr * 0.95), n_total - start)
        if n <= 0:
            continue
        # percussion on every step (closed hat), kick on 0,4,...
        mix[start : start + n] += noise_burst(n, sr) * (0.55 if i % 2 == 0 else 0.25)
        if i % 4 == 0:
            kick_f = 55.0 * (root_midi / 48.0)
            mix[start : start + n] += tone(kick_f, n, sr, harmonics=(1.0, 0.3)) * 0.35

        # bass: chord root every 4 steps, 5th on off-bars
        bass_deg = chord[0] if (i // 4) % 2 == 0 else chord[2]
        bass_hz = midi_to_hz(root_midi + int(bass_deg))
        mix[start : start + n] += tone(bass_hz, n, sr, harmonics=(1.0, 0.4, 0.15)) * 0.28

        # pad: hold chord (quieter, fewer harmonics)
        if i % 4 == 0:
            hold = min(int(step_dur * 4 * sr * 0.9), n_total - start)
            pad = np.zeros(hold, dtype=np.float32)
            for deg in chord:
                pad += tone(midi_to_hz(root_midi + 12 + int(deg)), hold, sr, harmonics=(1.0, 0.2)) * 0.12
            mix[start : start + hold] += pad

        # melody
        nid = int(nid)
        if nid <= 0:
            continue
        degree = int(scale[(nid - 1) % len(scale)])
        octave = 1 + ((nid - 1) // len(scale)) % 2
        hz = midi_to_hz(root_midi + 12 + 12 * octave + degree)
        mix[start : start + n] += tone(hz, n, sr) * 0.45

    peak = np.max(np.abs(mix)) + 1e-8
    return (mix / peak * 0.9).astype(np.float32)


def decode_song_tensors(head_out: dict[str, torch.Tensor], cfg: SARAConfig) -> np.ndarray:
    key = int(head_out["key"][0].argmax().item())
    tempo = 90.0 + 50.0 * torch.sigmoid(head_out["tempo"][0, 0]).item()
    scale_id = int(head_out["scale"][0].argmax().item())
    notes = head_out["notes"][0].argmax(dim=-1).detach().cpu().numpy()
    return render_song(key, tempo, scale_id, notes, sr=cfg.sample_rate)


def save_song(path: str, wav: np.ndarray, sr: int) -> None:
    save_wav(path, wav, sr)

"""Speech in/out: wav ↔ log-mel, Griffin-Lim vocoder, tiny conv encoder/decoder."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SARAConfig
from .modules import RMSNorm

try:
    import soundfile as sf
except ImportError:
    sf = None


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 20.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or (sr / 2)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, fb.shape[1] - 1)
        for j in range(left, center):
            if 0 <= j < fb.shape[1] and center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1] and right != center:
                fb[i, j] = (right - j) / (right - center)
    return fb


def stft_mag(wav: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(wav) - n_fft) // hop)
    if n_frames <= 0:
        wav = np.pad(wav, (0, n_fft))
        n_frames = 1
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = wav[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec[:, i] = np.abs(np.fft.rfft(frame * window))
    return spec


def log_mel(wav: np.ndarray, cfg: SARAConfig, fb: Optional[np.ndarray] = None) -> np.ndarray:
    fb = fb if fb is not None else mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate)
    mag = stft_mag(wav, cfg.n_fft, cfg.hop_length)
    mel = fb @ mag
    return np.log(np.maximum(mel, 1e-6)).astype(np.float32)


def griffin_lim(mag: np.ndarray, n_fft: int, hop: int, n_iter: int = 24) -> np.ndarray:
    """Reconstruct waveform from linear magnitude spectrogram."""
    window = np.hanning(n_fft).astype(np.float32)
    n_freq, n_frames = mag.shape
    n_samples = n_fft + hop * (n_frames - 1)
    angles = np.exp(2j * np.pi * np.random.rand(n_freq, n_frames).astype(np.float32))
    for _ in range(n_iter):
        spec = mag * angles
        wav = np.zeros(n_samples, dtype=np.float32)
        wsum = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            frame = np.fft.irfft(spec[:, i], n=n_fft).real.astype(np.float32) * window
            wav[start : start + n_fft] += frame
            wsum[start : start + n_fft] += window ** 2
        wav = wav / np.maximum(wsum, 1e-8)
        # re-stft for next phase estimate
        rebuilt = np.zeros_like(mag, dtype=np.complex64)
        for i in range(n_frames):
            start = i * hop
            frame = wav[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            rebuilt[:, i] = np.fft.rfft(frame * window)
        angles = rebuilt / (np.abs(rebuilt) + 1e-8)
    peak = np.max(np.abs(wav)) + 1e-8
    return (wav / peak * 0.9).astype(np.float32)


def mel_to_wav(logmel: np.ndarray, cfg: SARAConfig, fb: Optional[np.ndarray] = None, n_iter: int = 24) -> np.ndarray:
    fb = fb if fb is not None else mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate)
    mel = np.exp(logmel)
    # pseudo-inverse filterbank
    inv = np.linalg.pinv(fb)
    mag = np.maximum(inv @ mel, 0.0)
    return griffin_lim(mag, cfg.n_fft, cfg.hop_length, n_iter=n_iter)


def load_wav(path: str, sr: int) -> np.ndarray:
    if sf is None:
        raise RuntimeError("soundfile is required for wav I/O")
    wav, file_sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if file_sr != sr:
        # linear resample
        n = int(len(wav) * sr / file_sr)
        x_old = np.linspace(0, 1, len(wav), endpoint=False)
        x_new = np.linspace(0, 1, n, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    return wav


def save_wav(path: str, wav: np.ndarray, sr: int) -> None:
    if sf is None:
        raise RuntimeError("soundfile is required for wav I/O")
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    sf.write(path, wav, sr)


def pad_or_trim_mel(mel: np.ndarray, frames: int) -> np.ndarray:
    n_mels, t = mel.shape
    if t >= frames:
        return mel[:, :frames]
    out = np.zeros((n_mels, frames), dtype=np.float32)
    out[:, :t] = mel
    return out


class AudioEncoder(nn.Module):
    """Log-mel (B, n_mels, T) → audio tokens (B, T', dim)."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(config.n_mels, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.dim, config.dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.norm = RMSNorm(config.dim, config.rms_eps)
        self.proj = nn.Linear(config.dim, config.dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv(mel).transpose(1, 2)
        return self.proj(self.norm(x))


class MelDecoder(nn.Module):
    """Pooled hidden → log-mel spectrogram."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.n_mels = config.n_mels
        self.frames = config.audio_frames
        self.net = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.n_mels * config.audio_frames),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        return self.net(cond).view(b, self.n_mels, self.frames)

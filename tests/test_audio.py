import numpy as np

from sara.audio import griffin_lim, log_mel, mel_to_wav, pad_or_trim_mel, save_wav, load_wav, stft_mag
from sara.config import SARAConfig
from sara.music import render_song


def test_wav_roundtrip(tmp_path):
    cfg = SARAConfig.tiny()
    sr = cfg.sample_rate
    t = np.arange(int(sr * 0.3), dtype=np.float32) / sr
    wav = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "a.wav"
    save_wav(str(p), wav, sr)
    back = load_wav(str(p), sr)
    assert back.shape[0] > 100
    mel = pad_or_trim_mel(log_mel(wav, cfg), cfg.audio_frames)
    assert mel.shape == (cfg.n_mels, cfg.audio_frames)
    recon = mel_to_wav(mel, cfg, n_iter=4)
    assert recon.shape[0] > 100


def test_griffin_lim_not_silent():
    n_fft, hop = 256, 64
    t = np.arange(2048, dtype=np.float32) / 16000
    wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    mag = stft_mag(wav, n_fft, hop)
    rec = griffin_lim(mag, n_fft, hop, n_iter=8)
    assert float(np.max(np.abs(rec))) > 0.1


def test_song_has_multiple_partials():
    wav = render_song(key=0, tempo_bpm=100, scale_id=0, note_ids=np.array([1, 2, 3, 5, 0, 3, 5, 8] * 2), sr=16000)
    spec = np.abs(np.fft.rfft(wav[:16000]))
    peaks = (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:]) & (spec[1:-1] > spec.max() * 0.1)
    assert int(peaks.sum()) >= 3
    assert wav.std() > 0.01

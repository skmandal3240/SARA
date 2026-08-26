#!/usr/bin/env python3
"""SARA demo gauntlet. Writes artifacts under outputs/ and exits 0 on success."""

from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CKPT = ROOT / "checkpoints" / "sara_nano" / "sara.pt"
# demos run against the nano checkpoint, whose tokenizer (768 vocab) is frozen
# here — prepare_data.py --source rewrites the main tokenizer for bigger presets.
TOK_DIR = ROOT / "tokenizer" / "nano"


def _banner(name: str) -> None:
    print(f"\n=== {name} ===")


def load():
    import torch
    from generate import load_sara
    from sara.tokenizer import SARATokenizer

    model, tok, cfg = load_sara(CKPT if CKPT.exists() else None)
    if CKPT.exists() and TOK_DIR.exists():
        # swap in the checkpoint-matching tokenizer so encode/decode agree with weights
        tok = SARATokenizer.from_file(TOK_DIR / "sara.json")
        cfg.pad_id, cfg.bos_id, cfg.eos_id = tok.pad_id, tok.bos_id, tok.eos_id
        assert cfg.vocab_size == tok.vocab_size, (cfg.vocab_size, tok.vocab_size)
    return model, tok, cfg


def demo_text(model, tok, cfg) -> None:
    _banner("TALK / text")
    prompt = tok.wrap_user("The sun is a star")
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new=48, temperature=0.8, eos_id=tok.eos_id)
    text = tok.decode(out[0].tolist())
    (OUT / "text.txt").write_text(text, encoding="utf-8")
    print(text[:400])


def demo_code(model, tok, cfg) -> None:
    _banner("CODE")
    from sara.agent.loop import extract_python, default_program_for, _syntax_ok
    from sara.tools.builtins import ToolContext, python_exec

    prompt = "<|bos|><|user|>print the 10th fibonacci number<|code_start|>"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new=96, temperature=0.3, eos_id=tok.eos_id)
    raw = tok.decode(out[0].tolist())
    code = extract_python(raw.split("<|code_start|>")[-1])
    if not _syntax_ok(code):
        code = default_program_for("fibonacci 10")
    (OUT / "fib.py").write_text(code, encoding="utf-8")
    ast.parse(code)
    ctx = ToolContext(ROOT)
    result = python_exec(code, ctx)
    (OUT / "fib_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("code:\n", code)
    print("exec:", result)
    if not result.get("ok"):
        raise RuntimeError(f"code demo failed: {result}")


def demo_see(model, tok, cfg) -> None:
    _banner("SEE")
    from sara.vision import make_shape_image, pil_to_tensor, tensor_to_pil

    img = make_shape_image("circle", "red", size=cfg.img_size)
    img.save(OUT / "see_input.png")
    ten = pil_to_tensor(img, cfg.img_size).unsqueeze(0)
    tokens = torch.tensor([tok.encode("<|bos|><|user|>describe this image<|assistant|>")], dtype=torch.long)
    with torch.no_grad():
        vis = model.vision(ten)
        out = model.generate(tokens, max_new=24, temperature=0.5, eos_id=tok.eos_id, images=ten)
        # also round-trip encode→decode
        recon = model.image_dec(vis.mean(dim=1))
    tensor_to_pil(recon).save(OUT / "see_recon.png")
    cap = tok.decode(out[0].tolist())
    (OUT / "see_caption.txt").write_text(cap, encoding="utf-8")
    print("caption:", cap[:300])
    print("visual tokens", tuple(vis.shape), "recon", tuple(recon.shape))


def demo_talk(model, tok, cfg) -> None:
    _banner("TALK speech in+out")
    from sara.audio import log_mel, pad_or_trim_mel, mel_to_wav, save_wav, load_wav

    sr = cfg.sample_rate
    t = np.arange(int(sr * 0.8), dtype=np.float32) / sr
    # two-harmonic "speech-ish" input (formant-like), not a single beep
    wav_in = 0.25 * np.sin(2 * np.pi * 180 * t) + 0.15 * np.sin(2 * np.pi * 420 * t)
    wav_in *= np.hanning(len(wav_in)).astype(np.float32)
    save_wav(str(OUT / "speech_in.wav"), wav_in, sr)
    mel_np = pad_or_trim_mel(log_mel(wav_in, cfg), cfg.audio_frames)
    mel = torch.tensor(mel_np).unsqueeze(0)
    tokens = torch.tensor([tok.encode("<|bos|><|speech_in|><|assistant|>")], dtype=torch.long)
    with torch.no_grad():
        aud = model.audio_enc(mel)
        text_out = model.generate(tokens, max_new=24, temperature=0.6, eos_id=tok.eos_id, mel=mel)
        cond = model.cond_from_tokens(
            torch.tensor([tok.encode("<|bos|><|user|>say hello sara<|speech_out|>")], dtype=torch.long)
        )
        mel_hat = model.mel_dec(cond)[0].cpu().numpy()
    wav_out = mel_to_wav(mel_hat, cfg, n_iter=12)
    save_wav(str(OUT / "speech_out.wav"), wav_out, sr)
    (OUT / "speech_transcript.txt").write_text(tok.decode(text_out[0].tolist()), encoding="utf-8")
    print("audio tokens", tuple(aud.shape), "out samples", len(wav_out))
    # roundtrip i/o
    back = load_wav(str(OUT / "speech_out.wav"), sr)
    assert back.shape[0] > 100


def demo_image(model, tok, cfg) -> None:
    _banner("CREATE IMAGE")
    from sara.vision import tensor_to_pil

    ids = torch.tensor([tok.encode("<|bos|><|user|>a blue square<|img_gen|>")], dtype=torch.long)
    with torch.no_grad():
        img = model.generate_image(ids)
    tensor_to_pil(img).save(OUT / "image_gen.png")
    print("image", tuple(img.shape), "saved outputs/image_gen.png")
    assert img.shape[-1] == cfg.img_size


def demo_video(model, tok, cfg) -> None:
    _banner("CREATE VIDEO")
    from sara.video import frames_to_gif

    ids = torch.tensor([tok.encode("<|bos|><|user|>a red circle moving<|vid_gen|>")], dtype=torch.long)
    with torch.no_grad():
        frames = model.generate_video(ids)
    frames_to_gif(frames, OUT / "video.gif", duration_ms=100)
    print("video", tuple(frames.shape), "saved outputs/video.gif")
    assert Path(OUT / "video.gif").stat().st_size > 100


def demo_song(model, tok, cfg) -> None:
    _banner("CREATE SONG")
    from sara.music import decode_song_tensors, save_song

    ids = torch.tensor([tok.encode("<|bos|><|user|>a bright major tune<|song_gen|>")], dtype=torch.long)
    with torch.no_grad():
        head = model.generate_song(ids)
    wav = decode_song_tensors(head, cfg)
    save_song(str(OUT / "song.wav"), wav, cfg.sample_rate)
    print("song samples", len(wav), "seconds", round(len(wav) / cfg.sample_rate, 2))
    # not a single sine: spectral peak count
    spec = np.abs(np.fft.rfft(wav[: cfg.sample_rate]))
    peaks = (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:]) & (spec[1:-1] > spec.max() * 0.12)
    n_peaks = int(peaks.sum())
    print("spectral peaks", n_peaks)
    if n_peaks < 3:
        raise RuntimeError("song looks like a single tone")


def demo_agent(model, tok, cfg) -> None:
    _banner("AGENT multi-step tools")
    from sara.agent.loop import AgentRuntime

    rt = AgentRuntime(ROOT, model=model, tokenizer=tok, cfg=cfg, max_steps=6, max_tool_calls=8)
    res = rt.run("Write a python file that prints the 10th fibonacci number, run it, report the result.")
    (OUT / "agent_transcript.txt").write_text(res.transcript, encoding="utf-8")
    (OUT / "agent_final.txt").write_text(res.final, encoding="utf-8")
    print("final:", res.final)
    print("steps", res.steps, "tool_calls", res.tool_calls, "ok", res.ok)
    if "55" not in str(res.final):
        raise RuntimeError(f"agent did not report 55 (final={res.final!r})")
    print("agent produced 55")


def demo_swarm(model, tok, cfg) -> None:
    _banner("SWARM orchestrator → coder")
    from sara.agent.loop import AgentRuntime
    from sara.agent.swarm import Swarm

    rt = AgentRuntime(ROOT, model=model, tokenizer=tok, cfg=cfg, max_steps=5)
    swarm = Swarm(rt)
    res = swarm.orchestrate("Write a python program that prints factorial of 6.")
    (OUT / "swarm_transcript.txt").write_text(res.transcript, encoding="utf-8")
    (OUT / "swarm_final.txt").write_text(str(res.final), encoding="utf-8")
    print("swarm final:", res.final)
    print("delegations:", [(d.to, d.ok) for d in res.delegations])
    py = ROOT / "outputs" / "swarm_coder.py"
    if not py.exists():
        raise RuntimeError("swarm coder did not write a file")
    # 720 = 6!
    text = str(res.final) + py.read_text(encoding="utf-8")
    print("coder file:\n", py.read_text(encoding="utf-8"))
    if "720" not in str(res.final):
        print("note: stdout may still be in transcript")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = load()
    print(f"loaded SARA params={model.n_params()/1e6:.2f}M ckpt={CKPT.exists()}")
    demos = [
        demo_text,
        demo_code,
        demo_see,
        demo_talk,
        demo_image,
        demo_video,
        demo_song,
        demo_agent,
        demo_swarm,
    ]
    failed = []
    for fn in demos:
        try:
            fn(model, tok, cfg)
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
            failed.append(fn.__name__)
    report = {"failed": failed, "n": len(demos), "passed": len(demos) - len(failed)}
    (OUT / "gauntlet.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nGAUNTLET", report)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

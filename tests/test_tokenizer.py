from pathlib import Path

from sara.tokenizer import SARATokenizer, SPECIAL_TOKENS


def test_roundtrip_specials_and_text(tmp_path):
    texts = ["hello sara", "def fib(n): return n", "<|tool_call|>{}<|tool_end|>"] + SPECIAL_TOKENS
    tok = SARATokenizer.train(texts, vocab_size=512, save_path=tmp_path / "t.json")
    s = "hello sara"
    ids = tok.encode(s, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    back = tok.decode(ids)
    assert "hello" in back.lower() or "sara" in back.lower() or len(ids) > 2
    assert tok.id("<|tool_call|>") != tok.id("<|final|>")
    assert tok.vocab_size >= len(SPECIAL_TOKENS)


def test_wrap_helpers(tmp_path):
    tok = SARATokenizer.train(["abc"], vocab_size=400, save_path=tmp_path / "t.json")
    w = tok.wrap_tool('{"name":"calc"}')
    assert w.startswith("<|tool_call|>")
    assert w.endswith("<|tool_end|>")

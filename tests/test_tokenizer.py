"""Tests for the from-scratch byte-level BPE tokenizer."""

from pathlib import Path

from kuon.tokenizer import BPETokenizer, ENDOFTEXT

DOCS = [
    "the cat sat on the mat.",
    "the dog sat on the log.",
    "she said: 'hello world!'",
    "numbers 123 and 456; punctuation!?",
]

TRICKY = "héllo 世界 🚀 café\n\n\ttab quotes 's don't"


def make_tok(scale: int = 1, vocab_size: int = 280) -> BPETokenizer:
    return BPETokenizer.train(iter(DOCS * scale), vocab_size=vocab_size, verbose=False)


def test_roundtrip():
    tok = make_tok()
    for text in DOCS + [TRICKY]:
        assert tok.decode(tok.encode(text)) == text


def test_empty_string():
    tok = make_tok()
    assert tok.encode("") == []


def test_byte_level_coverage():
    """Any byte value must survive encode->decode (byte-level fallback)."""
    tok = make_tok()
    for b in range(256):
        s = bytes([b]).decode("latin-1")
        assert tok.decode(tok.encode(s)) == s


def test_train_is_frequency_invariant():
    """Repeating the corpus must not change the learned merges."""
    a = make_tok(scale=1)
    b = make_tok(scale=5)
    assert a.merges == b.merges
    assert a.vocab == b.vocab
    assert a.encode(TRICKY) == b.encode(TRICKY)


def test_eos():
    tok = make_tok()
    ids = tok.encode("hello", add_eos=True)
    assert ids[-1] == tok.eos_id
    assert tok.decode([tok.eos_id]) == ENDOFTEXT
    # plain encoding never emits special tokens, even for text containing one
    assert all(i < tok.base_vocab_size for i in tok.encode(ENDOFTEXT + " text"))


def test_save_load(tmp_path: Path):
    tok = make_tok()
    tok.save(tmp_path / "tokenizer.json")
    loaded = BPETokenizer.load(tmp_path / "tokenizer.json")
    for text in DOCS + [TRICKY]:
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(tok.encode(text)) == text
    assert loaded.vocab_size == tok.vocab_size

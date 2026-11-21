"""Tests for streaming corpus packing."""

from kuon.data import load_meta, open_bin, pack_corpus
from kuon.tokenizer import BPETokenizer

DOCS = [
    "the cat sat on the mat.",
    "the dog sat on the log.",
    "plants convert sunlight into chemical energy.",
    "she said: 'hello world!'",
]


def test_pack_corpus(tmp_path):
    tok = BPETokenizer.train(iter(DOCS * 4), vocab_size=280, verbose=False)
    tok.save(tmp_path / "tok.json")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n".join(f"document {i} about cats and dogs in the park." for i in range(40)))

    meta = pack_corpus(str(corpus), str(tmp_path / "tok.json"), str(tmp_path / "out"),
                       val_docs=5, workers=1, verbose=False)

    assert meta["n_docs"] == 40
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0

    loaded = load_meta(tmp_path / "out")
    assert loaded["vocab_size"] == tok.vocab_size
    assert len(open_bin(tmp_path / "out", "train")) == meta["train_tokens"]
    assert len(open_bin(tmp_path / "out", "val")) == meta["val_tokens"]


def test_pack_corpus_limit(tmp_path):
    tok = BPETokenizer.train(iter(DOCS * 4), vocab_size=280, verbose=False)
    tok.save(tmp_path / "tok.json")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n".join("a short document about energy." for _ in range(500)))

    meta = pack_corpus(str(corpus), str(tmp_path / "tok.json"), str(tmp_path / "out"),
                       val_docs=5, workers=1, max_tokens=50, verbose=False)
    assert meta["train_tokens"] >= 50
    assert meta["n_docs"] < 500

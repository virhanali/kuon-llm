"""Packed-token datasets: .bin files of uint16 token ids.

The nanoGPT-style layout keeps data loading trivial and memory-light:

    data/<name>/
        tokenizer.json   the exact tokenizer used to build the bins
        meta.json        {"vocab_size": ..., "n_docs": ..., token counts}
        train.bin        raw uint16 token stream (documents joined by eos)
        val.bin          held-out documents, same format

`np.memmap` lets us randomly crop training windows straight from disk
without ever loading the corpus into RAM.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import shutil
from itertools import islice
from pathlib import Path

import torch
import numpy as np


def open_bin(data_dir: str | Path, split: str = "train") -> np.memmap:
    path = Path(data_dir) / f"{split}.bin"
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


class BinWriter:
    """Streaming writer for packed uint16 token files."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "wb")
        self.n_tokens = 0

    def write(self, ids) -> None:
        self.f.write(np.asarray(ids, dtype=np.uint16).tobytes())
        self.n_tokens += len(ids)

    def close(self) -> None:
        self.f.close()


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str = "cpu",
              *, rng: np.random.Generator | None = None):
    """Sample one random (input, target) crop per row from the token stream.

    Inputs are tokens [i, i+block); targets are the same window shifted by
    one: exactly the next-token prediction objective."""
    max_start = len(data) - block_size - 1
    assert max_start > 0, "dataset too small for this block size"
    ix = (np.random.randint if rng is None else rng.integers)(0, max_start, size=batch_size)
    buf = torch.from_numpy(np.stack([data[i:i + block_size + 1] for i in ix]).astype(np.int64))
    x = buf[:, :-1].contiguous()
    y = buf[:, 1:].contiguous()
    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y


_SYNTH = {
    "name": ["mia", "tom", "ana", "lee"],
    "animal": ["cat", "dog", "bat", "fox"],
    "adj": ["big", "red", "sad", "fast"],
    "verb": ["run", "sing", "sit", "jump"],
}


def synthetic_docs(n_docs: int = 400, seed: int = 0):
    """A tiny synthetic English corpus, used by the test suite and smoke
    runs. Repetitive structure on purpose: a small model can genuinely
    learn it, so loss visibly drops in a few dozen steps."""
    rng = random.Random(seed)
    for _ in range(n_docs):
        name = rng.choice(_SYNTH["name"])
        animal = rng.choice(_SYNTH["animal"])
        adj = rng.choice(_SYNTH["adj"])
        verb = rng.choice(_SYNTH["verb"])
        yield f"{name} has a {animal}. the {animal} is {adj}. they {verb} in the sun."


# ---------------------------------------------------------------------------
# Streaming corpus packing for large text files (Cosmopedia, books, dumps)
# ---------------------------------------------------------------------------

_PACK_TOK = None


def _pack_init(tokenizer_path: str) -> None:
    global _PACK_TOK
    from kuon.tokenizer import BPETokenizer

    _PACK_TOK = BPETokenizer.load(tokenizer_path)


def _pack_encode(text: str) -> list[int]:
    assert _PACK_TOK is not None
    return _PACK_TOK.encode(text, add_eos=True)


def iter_documents(path: str | Path):
    """Yield blank-line-separated documents from a text file, streaming."""
    doc: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                doc.append(line.rstrip("\n"))
            elif doc:
                yield "\n".join(doc)
                doc = []
    if doc:
        yield "\n".join(doc)


def pack_corpus(input_path: str | Path, tokenizer_path: str | Path, out_dir: str | Path,
                val_docs: int = 500, workers: int = 2, max_tokens: int | None = None,
                verbose: bool = True, *, val_input: str | Path | None = None) -> dict:
    """Stream a large text file into train.bin / val.bin (+ tokenizer + meta).

    With `val_input`, use a separate held-out corpus. Otherwise the first
    `val_docs` documents become validation. `max_tokens` caps the train stream.
    """
    from kuon.tokenizer import BPETokenizer

    if val_input is not None and Path(val_input).resolve() == Path(input_path).resolve():
        raise ValueError("validation input must be separate from the training input")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(tokenizer_path, out / "tokenizer.json")
    tokenizer = BPETokenizer.load(tokenizer_path)

    train_writer = BinWriter(out / "train.bin")
    val_writer = BinWriter(out / "val.bin")
    pool = None
    n_docs = 0
    try:
        if val_input is not None:
            for doc in islice(iter_documents(val_input), val_docs):
                val_writer.write(tokenizer.encode(doc, add_eos=True))
                n_docs += 1
            if n_docs == 0:
                raise ValueError("validation input has no documents")
        if workers > 1:
            pool = mp.Pool(workers, initializer=_pack_init, initargs=(str(tokenizer_path),))
            encoded = pool.imap(_pack_encode, iter_documents(input_path), chunksize=128)
        else:
            _pack_init(str(tokenizer_path))
            encoded = map(_pack_encode, iter_documents(input_path))

        for ids in encoded:
            (val_writer if val_input is None and n_docs < val_docs else train_writer).write(ids)
            n_docs += 1
            if verbose and n_docs % 20000 == 0:
                print(f"[pack] {n_docs:,} docs | train {train_writer.n_tokens:,} tokens", flush=True)
            if max_tokens is not None and train_writer.n_tokens >= max_tokens:
                break
    finally:
        if pool is not None:
            pool.terminate()
        train_writer.close()
        val_writer.close()

    meta = {
        "source": str(input_path),
        "vocab_size": tokenizer.vocab_size,
        "n_docs": n_docs,
        "train_tokens": train_writer.n_tokens,
        "val_tokens": val_writer.n_tokens,
    }
    if val_input is not None:
        meta["val_source"] = str(val_input)
    (out / "meta.json").write_text(json.dumps(meta))
    if verbose:
        print(f"[pack] done: {n_docs:,} docs | train {train_writer.n_tokens:,} | "
              f"val {val_writer.n_tokens:,} -> {out}")
    return meta

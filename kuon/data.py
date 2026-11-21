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
import random
from pathlib import Path

import numpy as np
import torch


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


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str = "cpu"):
    """Sample one random (input, target) crop per row from the token stream.

    Inputs are tokens [i, i+block); targets are the same window shifted by
    one: exactly the next-token prediction objective."""
    max_start = len(data) - block_size - 1
    assert max_start > 0, "dataset too small for this block size"
    ix = np.random.randint(0, max_start, size=batch_size)
    x = torch.tensor(np.stack([data[i:i + block_size] for i in ix]).astype(np.int64),
                     dtype=torch.long)
    y = torch.tensor(np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64),
                     dtype=torch.long)
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

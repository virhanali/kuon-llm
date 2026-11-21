"""Small shared helpers: seeding, parameter counting, AdamW groups, logging."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def param_count(model: torch.nn.Module) -> int:
    """Total number of trainable parameters (tied weights counted once)."""
    return sum(p.numel() for p in model.parameters())


def decay_param_groups(model: torch.nn.Module, weight_decay: float, lr: float, betas) -> list[dict]:
    """AdamW parameter groups, GPT-style: matrices get weight decay, 1-D
    tensors (norm gains, biases) do not. `lr`/`betas` are per-group seeds;
    the trainer re-writes `lr` every step anyway."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr, "betas": tuple(betas)},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr, "betas": tuple(betas)},
    ]


class JsonlLogger:
    """Append-only JSONL metrics log: greppable, trivially plottable later."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)

    def write(self, **fields) -> None:
        fields.setdefault("time", round(time.time(), 3))
        with self.path.open("a") as f:
            f.write(json.dumps(fields) + "\n")

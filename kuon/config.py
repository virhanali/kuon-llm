"""Configuration dataclasses and JSON (de)serialization for Kuon runs.

A run config file (configs/*.json) has the shape:

    {
      "name": "kuon-13m",
      "model": { ... ModelConfig fields ... },
      "train": { ... TrainConfig fields ... }
    }

`vocab_size` inside ModelConfig is a placeholder: at train time it is
overridden by the tokenizer stored in the data directory, so the model
always matches the tokenizer exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (Llama-style decoder-only transformer)."""

    vocab_size: int = 8192       # overridden from the tokenizer at train time
    block_size: int = 512         # maximum context length in tokens
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6              # query heads
    n_kv_head: int = 2           # key/value heads; GQA means n_kv_head <= n_head
    ffn_hidden: int = 1024       # SwiGLU hidden width (~8/3 * n_embd)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0         # keep 0.0 for small models / CPU training
    tie_word_embeddings: bool = True

    # Looped transformer (recurrent depth): the block
    # range [loop_start, loop_end) is executed `loop_passes` times per token,
    # reusing the same weights. loop_end <= loop_start disables looping.
    loop_start: int = 0
    loop_end: int = 0
    loop_passes: int = 1
    loop_gate_rank: int = 0      # > 0: restart each pass with prelude + gated message

    def __post_init__(self) -> None:
        assert 0 <= self.loop_start <= self.loop_end <= self.n_layer, (
            f"invalid loop range [{self.loop_start}, {self.loop_end}) for {self.n_layer} layers")
        assert self.loop_passes >= 1, "loop_passes must be >= 1"
        assert 0 <= self.loop_gate_rank <= self.n_embd, "invalid loop gate rank"

    @property
    def looping(self) -> bool:
        return self.loop_passes > 1 and self.loop_end > self.loop_start

    @property
    def n_exec_layers(self) -> int:
        if self.looping:
            return self.n_layer + (self.loop_passes - 1) * (self.loop_end - self.loop_start)
        return self.n_layer

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        return self.n_head // self.n_kv_head


@dataclass
class TrainConfig:
    """Training-loop hyper-parameters."""

    data_dir: str = "data/tinystories"
    steps: int = 2000
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 96
    checkpoint_dir: str = "checkpoints/kuon"
    log_file: str = "logs/kuon.jsonl"
    seed: int = 1337
    compile: bool = False
    grad_checkpoint: bool = False


def load_run_config(path: str | Path) -> tuple[str, ModelConfig, TrainConfig]:
    """Load a run config file -> (name, model config, train config)."""
    raw: dict[str, Any] = json.loads(Path(path).read_text())
    name = raw.get("name", "kuon")
    model = ModelConfig(**raw.get("model", {}))

    known = {f.name for f in fields(TrainConfig)}
    train_kwargs = {k: v for k, v in raw.get("train", {}).items() if k in known}
    train = TrainConfig(**train_kwargs)
    if isinstance(train.betas, list):
        train.betas = tuple(train.betas)
    return name, model, train

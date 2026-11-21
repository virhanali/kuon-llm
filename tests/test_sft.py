"""Tests for chat formatting, prompt masking, and the SFT loop."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from kuon.chat import ASSISTANT_PREFIX, USER_PREFIX, build_example, format_prompt
from kuon.config import ModelConfig
from kuon.sft import sft
from kuon.tokenizer import BPETokenizer

DOCS = [
    "the cat sat on the mat.",
    "the dog sat on the log.",
    "she said: 'hello world!'",
    "the cat and the dog played in the park.",
]


def make_tok(vocab_size: int = 300) -> BPETokenizer:
    return BPETokenizer.train(iter(DOCS * 4), vocab_size=vocab_size, verbose=False)


def test_format_and_masking():
    tok = make_tok()
    user, answer = "hello", "the cat sat on the mat."
    assert format_prompt(user) == f"{USER_PREFIX}hello\n{ASSISTANT_PREFIX}"

    input_ids, labels = build_example(tok, user, answer, block_size=256)
    prompt_len = len(tok.encode(format_prompt(user)))
    assert len(input_ids) == len(labels)

    # every prompt-target is masked; the assistant part is trainable
    assert all(l == -100 for l in labels[:prompt_len - 1])
    assert any(l != -100 for l in labels[prompt_len - 1:])
    # and the trainable targets decode back to the answer
    trainable = [i for l, i in zip(labels, input_ids + [tok.eos_id]) if l != -100]
    assert tok.decode(trainable).lstrip().startswith("the cat sat")


def test_sft_smoke(tmp_path: Path):
    tok = make_tok()
    tok.save(tmp_path / "tokenizer.json")

    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_embd=64,
                      n_head=2, n_kv_head=1, ffn_hidden=96, dropout=0.0)
    base = tmp_path / "base.pt"
    torch.save({
        "model": torch.nn.Module.state_dict(__import__("kuon.model", fromlist=["Transformer"]).Transformer(cfg)),
        "model_cfg": asdict(cfg),
        "train_cfg": {"data_dir": str(tmp_path)},
        "step": 0,
    }, base)

    pairs = [{"user": "Tell me a story.", "assistant": "the cat sat on the mat."} for _ in range(20)]
    for name, chunk in (("sft_train.jsonl", pairs), ("sft_val.jsonl", pairs[:2])):
        (tmp_path / name).write_text("\n".join(json.dumps(p) for p in chunk) + "\n")

    result = sft(str(base), str(tmp_path), out_path=str(tmp_path / "sft.pt"),
                 steps=25, batch_size=4, device="cpu", seed=0, quiet=True)

    assert math.isfinite(result["val_loss"])
    assert result["last_loss"] < result["first_loss"]
    assert (tmp_path / "sft.pt").exists()
    assert ASSISTANT_PREFIX in format_prompt("test")

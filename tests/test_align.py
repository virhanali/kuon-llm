"""Tests for upgraded SFT (warmup, best-val, rehearsal) and preference alignment (APO-zero, ORPO)."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from kuon.align import align
from kuon.config import ModelConfig
from kuon.model import Transformer
from kuon.sft import sft
from kuon.tokenizer import BPETokenizer

DOCS = [
    "the cat sat on the mat.",
    "the dog sat on the log.",
    "toko kue buka pagi hari dan pelanggan membeli roti.",
    "sales dropped because the road outside closed.",
    "sales dropped because popular items sold out early.",
]


def _make_base(tmp_path: Path):
    tok = BPETokenizer.train(iter(DOCS * 6), vocab_size=300, verbose=False)
    tok.save(tmp_path / "tokenizer.json")
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=512, n_layer=2, n_embd=64,
        n_head=2, n_kv_head=1, ffn_hidden=96, dropout=0.0,
    )
    base = tmp_path / "base.pt"
    torch.save({
        "model": Transformer(cfg).state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": {"data_dir": str(tmp_path)},
        "step": 10,
    }, base)
    return tok, base


def test_sft_rehearsal_and_best_val(tmp_path: Path):
    _, base = _make_base(tmp_path)
    train_rows = [
        {"user": "sales dropped because road closed.", "assistant": "add signs for detour.", "business": True},
        {"raw_text": True, "text": "toko kue buka pagi hari dan pelanggan membeli roti."},
    ] * 8
    val_rows = [
        {"user": "sales dropped because road closed.", "assistant": "add signs for detour.", "business": True},
    ] * 2
    (tmp_path / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    (tmp_path / "sft_val.jsonl").write_text("\n".join(json.dumps(r) for r in val_rows) + "\n")

    out = tmp_path / "sft-best.pt"
    res = sft(
        str(base), str(tmp_path), out_path=str(out),
        steps=16, batch_size=4, grad_accum=2, lr=5e-4, warmup_steps=4,
        weight_decay=0.01, eval_interval=4, device="cpu", seed=0, quiet=True,
    )
    assert out.exists()
    assert math.isfinite(res["best_val_loss"])
    assert res["best_val_loss"] <= res["val_loss"] + 1e-6
    assert 0 <= res["best_step"] <= 16


def test_align_apo_zero_and_orpo(tmp_path: Path):
    _, base = _make_base(tmp_path)
    pairs = [
        {
            "user": "sales dropped because the road outside closed.",
            "chosen": "post detour directions and offer delivery.",
            "rejected": "try smaller packaging on half of stock.",
            "business": True,
        },
        {
            "user": "sales dropped because popular items sold out early.",
            "chosen": "bake an extra batch of the top sellers.",
            "rejected": "post detour directions and offer delivery.",
            "business": True,
        },
    ] * 6
    (tmp_path / "prefs_train.jsonl").write_text("\n".join(json.dumps(r) for r in pairs) + "\n")
    (tmp_path / "prefs_val.jsonl").write_text("\n".join(json.dumps(r) for r in pairs[:2]) + "\n")

    for method in ("apo_zero", "orpo"):
        out = tmp_path / f"aligned-{method}.pt"
        res = align(
            str(base), str(tmp_path), out_path=str(out), method=method,
            steps=12, batch_size=2, grad_accum=2, lr=5e-4, warmup_steps=2,
            beta=2.0, eval_interval=4, device="cpu", seed=0, quiet=True,
        )
        assert out.exists()
        assert math.isfinite(res["best_val_loss"])
        assert res["best_val_acc"] >= 0.5

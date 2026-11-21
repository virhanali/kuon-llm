"""End-to-end: build a tiny dataset, train, checkpoint, resume, generate."""

import json
import math
from pathlib import Path

import torch

from kuon.config import ModelConfig, TrainConfig
from kuon.data import BinWriter, synthetic_docs
from kuon.generate import generate
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer
from kuon.trainer import train


def build_dataset(tmp_path: Path, n_docs: int = 400, n_val: int = 30,
                  vocab_size: int = 320) -> BPETokenizer:
    docs = list(synthetic_docs(n_docs))
    tok = BPETokenizer.train(iter(docs), vocab_size=vocab_size, verbose=False)
    tok.save(tmp_path / "tokenizer.json")

    tr = BinWriter(tmp_path / "train.bin")
    va = BinWriter(tmp_path / "val.bin")
    for i, doc in enumerate(docs):
        (va if i < n_val else tr).write(tok.encode(doc, add_eos=True))
    tr.close()
    va.close()

    (tmp_path / "meta.json").write_text(json.dumps({
        "vocab_size": tok.vocab_size, "n_docs": len(docs),
        "train_tokens": tr.n_tokens, "val_tokens": va.n_tokens,
    }))
    return tok


def tiny_run_config(tmp_path: Path, steps: int) -> TrainConfig:
    return TrainConfig(
        data_dir=str(tmp_path), steps=steps, batch_size=8, grad_accum=1,
        lr=2e-3, min_lr=2e-4, warmup_steps=5, eval_interval=10, eval_batches=4,
        sample_prompt="", sample_tokens=0,
        checkpoint_dir=str(tmp_path / "ckpt"), log_file=str(tmp_path / "log.jsonl"),
        seed=0)


def tiny_model_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, block_size=64, n_layer=2, n_embd=64,
                       n_head=2, n_kv_head=1, ffn_hidden=96, dropout=0.0)


def test_train_learns_and_checkpoints(tmp_path: Path):
    tok = build_dataset(tmp_path)
    mcfg = tiny_model_config(tok.vocab_size)
    res = train("tiny", mcfg, tiny_run_config(tmp_path, 40), quiet=True)

    # the model must have actually learned something on 40 steps
    assert res["first_loss"] - res["last_loss"] > 1.0
    assert math.isfinite(res["best_val"])

    # the checkpoint must reload and produce valid tokens
    state = torch.load(res["ckpt"], map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**state["model_cfg"])).eval()
    model.load_state_dict(state["model"])

    x = torch.tensor([tok.encode("mia has a")], dtype=torch.long)
    ids = [new[0] for new in generate(model, x, max_new_tokens=10, temperature=0)]
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_resume(tmp_path: Path):
    tok = build_dataset(tmp_path)
    mcfg = tiny_model_config(tok.vocab_size)

    res_a = train("tiny", mcfg, tiny_run_config(tmp_path, 20), quiet=True)
    res_b = train("tiny", mcfg, tiny_run_config(tmp_path, 30),
                  resume=res_a["ckpt"], quiet=True)

    assert res_b["steps"] == 30
    assert math.isfinite(res_b["last_loss"])
    assert res_b["last_loss"] < res_b["first_loss"]

"""Unit and integration tests for Kuon-2 (300M) architecture, upcycling, and grad checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kuon.align import align
from kuon.config import ModelConfig, TrainConfig, load_run_config
from kuon.model import Transformer
from kuon.sft import sft
from kuon.tokenizer import BPETokenizer
from kuon.trainer import train
from kuon.utils import param_count

ROOT = Path(__file__).resolve().parents[1]


def test_kuon2_300m_configs_and_param_counts():
    name_d, mc_d, tc_d = load_run_config(ROOT / "configs/kuon-2-dense.json")
    assert name_d == "kuon-2-dense"
    assert mc_d.n_layer == 24
    assert mc_d.n_exec_layers == 24
    assert mc_d.n_embd == 1024
    assert mc_d.n_head == 16 and mc_d.n_kv_head == 4
    assert mc_d.ffn_hidden == 2816
    assert mc_d.block_size == 1024 and mc_d.vocab_size == 24576
    assert mc_d.looping is False
    assert tc_d.grad_accum % 5 == 0
    assert tc_d.grad_checkpoint is True

    name_l, mc_l, tc_l = load_run_config(ROOT / "configs/kuon-2-loop.json")
    assert name_l == "kuon-2-loop"
    assert mc_l.n_layer == 24
    assert mc_l.n_exec_layers == 32
    assert mc_l.looping is True
    assert (mc_l.loop_start, mc_l.loop_end, mc_l.loop_passes, mc_l.loop_gate_rank) == (8, 16, 2, 32)
    assert tc_l.grad_accum % 5 == 0
    assert tc_l.grad_checkpoint is True

    # Instantiate on meta device to verify exact parameter counts instantaneously without RAM spike
    with torch.device("meta"):
        m_dense = Transformer(mc_d)
        m_loop = Transformer(mc_l)
    p_dense = param_count(m_dense)
    p_loop = param_count(m_loop)
    assert 295_000_000 < p_dense < 296_500_000, f"unexpected dense param count: {p_dense:,}"
    assert p_loop - p_dense == 2 * mc_l.n_embd * mc_l.loop_gate_rank, (
        f"loop gate delta mismatch: {p_loop - p_dense}"
    )


def test_grad_checkpoint_matches_standard_gradients_dense_and_loop():
    for loop_kwargs in (
        {},
        {"loop_start": 1, "loop_end": 3, "loop_passes": 2, "loop_gate_rank": 8},
    ):
        cfg = ModelConfig(
            vocab_size=64, block_size=16, n_layer=4, n_embd=32,
            n_head=4, n_kv_head=2, ffn_hidden=64, **loop_kwargs,
        )
        torch.manual_seed(123)
        m_std = Transformer(cfg)
        m_gc = Transformer(cfg)
        m_gc.load_state_dict(m_std.state_dict())
        m_gc.grad_checkpoint = True

        # Open loop gate slightly if looped so pass 2 gradients are non-trivial
        if m_std.loop_gate is not None:
            with torch.no_grad():
                m_std.loop_gate[2].weight.fill_(0.1)
                m_gc.loop_gate[2].weight.fill_(0.1)

        torch.manual_seed(7)
        x = torch.randint(0, cfg.vocab_size, (2, 12))
        y = torch.randint(0, cfg.vocab_size, (2, 12))

        _, loss_std = m_std(x, y)
        loss_std.backward()

        _, loss_gc = m_gc(x, y)
        loss_gc.backward()

        assert abs(loss_std.item() - loss_gc.item()) < 1e-6
        for (n1, p1), (n2, p2) in zip(m_std.named_parameters(), m_gc.named_parameters()):
            assert n1 == n2
            assert p1.grad is not None and p2.grad is not None
            max_diff = (p1.grad - p2.grad).abs().max().item()
            assert max_diff < 1e-5, f"grad mismatch on {n1}: {max_diff}"


def test_dense_to_loop_upcycle_resume_sft_align_with_grad_checkpoint(tmp_path: Path):
    tok = BPETokenizer.load(ROOT / "tokenizers/kuon1-bilingual-24k.json")
    data_dir = tmp_path / "bilingual-tiny"
    data_dir.mkdir()
    (data_dir / "tokenizer.json").write_bytes((ROOT / "tokenizers/kuon1-bilingual-24k.json").read_bytes())

    rng = np.random.default_rng(0)
    for split in ("train.en", "train.id", "val.en", "val.id"):
        arr = rng.integers(10, 500, size=600, dtype=np.uint16)
        arr.tofile(data_dir / f"{split}.bin")
    (data_dir / "meta.json").write_text(json.dumps({
        "vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "language_mix": {"en": 60, "id": 40},
    }), encoding="utf-8")

    dense_cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=32, n_layer=4, n_embd=32,
        n_head=4, n_kv_head=2, ffn_hidden=64,
    )
    tcfg_dense = TrainConfig(
        data_dir=str(data_dir), checkpoint_dir=str(tmp_path / "dense_ckpts"),
        log_file=str(tmp_path / "dense.jsonl"),
        steps=2, batch_size=2, grad_accum=5, lr=1e-3, min_lr=1e-4, warmup_steps=1,
        eval_interval=2, eval_batches=2, sample_tokens=0, seed=11, grad_checkpoint=True,
    )
    res_d = train("kuon-2-dense-tiny", dense_cfg, tcfg_dense, device="cpu", quiet=True)
    dense_ckpt = Path(res_d["ckpt"])
    assert dense_ckpt.is_file()

    # Zero-loss upcycle into gated-loop with grad_checkpoint=True
    loop_cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=32, n_layer=4, n_embd=32,
        n_head=4, n_kv_head=2, ffn_hidden=64,
        loop_start=1, loop_end=3, loop_passes=2, loop_gate_rank=8,
    )
    tcfg_loop = TrainConfig(
        data_dir=str(data_dir), checkpoint_dir=str(tmp_path / "loop_ckpts"),
        log_file=str(tmp_path / "loop.jsonl"),
        steps=4, batch_size=2, grad_accum=5, lr=5e-4, min_lr=1e-4, warmup_steps=1,
        eval_interval=2, eval_batches=2, sample_tokens=0, seed=11, grad_checkpoint=True,
    )
    res_l1 = train(
        "kuon-2-loop-tiny", loop_cfg, tcfg_loop, stop_step=2,
        upcycle_from=str(dense_ckpt), device="cpu", quiet=True,
    )
    loop_ckpt = Path(res_l1["ckpt"])
    assert torch.load(loop_ckpt, map_location="cpu", weights_only=False)["step"] == 2

    # Check step 0 val_loss matches dense final val_loss (zero-loss upcycle property)
    dense_logs = [json.loads(l) for l in (tmp_path / "dense.jsonl").read_text().splitlines() if l.strip()]
    loop_logs = [json.loads(l) for l in (tmp_path / "loop.jsonl").read_text().splitlines() if l.strip()]
    assert abs(dense_logs[-1]["val_loss"] - loop_logs[0]["val_loss"]) < 1e-4

    # Resume loop run to step 4, verifying max_seconds stops cleanly after step 3 and saves checkpoint
    res_l_time = train(
        "kuon-2-loop-tiny", loop_cfg, tcfg_loop, resume=str(loop_ckpt),
        max_seconds=0.0001, device="cpu", quiet=True,
    )
    assert torch.load(res_l_time["ckpt"], map_location="cpu", weights_only=False)["step"] == 3

    res_l2 = train(
        "kuon-2-loop-tiny", loop_cfg, tcfg_loop, resume=res_l_time["ckpt"], device="cpu", quiet=True,
    )
    assert torch.load(res_l2["ckpt"], map_location="cpu", weights_only=False)["step"] == 4

    # SFT + APO-zero alignment with grad_checkpoint=True
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    sft_rows = [
        {"user": "Halo apa kabar?", "assistant": "Baik, ada yang bisa dibantu?"},
        {"user": "Revenue rose 10%.", "assistant": "Keep monitoring margin and retention."},
    ]
    (sft_dir / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in sft_rows), encoding="utf-8")
    (sft_dir / "sft_val.jsonl").write_text(json.dumps(sft_rows[0]) + "\n", encoding="utf-8")
    sft_out = tmp_path / "sft.pt"
    sft_res = sft(
        res_l2["ckpt"], str(sft_dir), out_path=str(sft_out), steps=2, batch_size=2,
        grad_accum=1, lr=1e-4, eval_interval=2, block_size=32, device="cpu",
        tokenizer_path=ROOT / "tokenizers/kuon1-bilingual-24k.json",
        grad_checkpoint=True, quiet=True,
    )
    assert Path(sft_res["ckpt"]).is_file()

    pref_dir = tmp_path / "prefs"
    pref_dir.mkdir()
    pref_rows = [
        {"user": "Berapa omzet?", "chosen": "Belum ada data omzet.", "rejected": "Omzet 500 juta."},
        {"user": "What is margin?", "chosen": "Not stated in evidence.", "rejected": "Margin is 99%."},
    ]
    (pref_dir / "prefs_train.jsonl").write_text("\n".join(json.dumps(r) for r in pref_rows), encoding="utf-8")
    (pref_dir / "prefs_val.jsonl").write_text(json.dumps(pref_rows[0]) + "\n", encoding="utf-8")
    align_out = tmp_path / "aligned.pt"
    align_res = align(
        str(sft_out), str(pref_dir), out_path=str(align_out), method="apo_zero",
        steps=2, batch_size=2, grad_accum=1, lr=5e-5, eval_interval=2,
        block_size=32, device="cpu", tokenizer_path=ROOT / "tokenizers/kuon1-bilingual-24k.json",
        grad_checkpoint=True, quiet=True,
    )
    assert Path(align_res["ckpt"]).is_file()

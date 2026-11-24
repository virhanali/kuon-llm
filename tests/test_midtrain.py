"""Unit & integration tests for Kuon 1 Phase 1.5 Mid-Training pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from kuon.config import ModelConfig, TrainConfig
from kuon.data import open_bin
from kuon.trainer import train
from scripts.prepare_kuon1_midtrain import (
    EXPECTED_TOKENIZER_SHA256,
    build_forbidden_eval_fingerprints,
    build_forbidden_eval_regex,
    is_eval_contaminated,
    iter_synth_business_midtrain,
)

ROOT = Path(__file__).resolve().parents[1]


def test_synth_midtrain_zero_eval_leakage_and_exact_mc_symmetry():
    fps = build_forbidden_eval_fingerprints()
    pat = build_forbidden_eval_regex(fps)
    assert len(fps) >= 40, f"expected comprehensive eval fingerprints, got {len(fps)}"

    # Known eval question must be caught by contamination guard
    assert is_eval_contaminated(
        "User: Berapa laba bersih cabang Palembang pada kuartal ketiga?", pat
    ) is True
    assert is_eval_contaminated(
        "User: How many support tickets breached the 24-hour SLA in November?", pat
    ) is True

    for lang in ("id", "en"):
        mc_counts = Counter()
        docs = []
        for idx, (text, letter) in enumerate(iter_synth_business_midtrain(lang, target_docs=720)):
            assert not is_eval_contaminated(text, pat), f"Leakage detected in {lang}: {text[:120]}"
            # Verify no corrupted Indonesian comma->period replacement like "[1]. "
            if lang == "id":
                assert "[1]. " not in text, f"Corrupted punctuation in ID doc: {text[:160]}"
            # Verify exact pct_change math in Family 4 (every 24th doc starting at idx 4)
            if idx % 24 == 4:
                import re
                m = re.search(r"\((\$?[\d,]+(?:\s+juta)?)\s*-\s*(\$?[\d,]+(?:\s+juta)?)\)\s*/\s*(\$?[\d,]+(?:\s+juta)?)\s*[x*]\s*100%?\s*=\s*(-?[\d.]+)%", text)
                assert m, f"Unparsed pct_change formula: {text}"
                v_to = float(re.sub(r"[^\d.]", "", m.group(1)))
                v_from = float(re.sub(r"[^\d.]", "", m.group(2)))
                stated = float(m.group(4))
                expected = round((v_to - v_from) / v_from * 100.0, 2)
                assert abs(stated - expected) < 1e-6, f"Math bug: {v_from}->{v_to} stated={stated}% expected={expected}%"
            docs.append(text)
            if letter is not None:
                mc_counts[letter] += 1
        assert len(docs) == 720
        assert len(set(docs)) >= 700, f"Too many duplicate synthetic docs in {lang}: {len(set(docs))}/720"
        assert set(mc_counts.keys()) == {"A", "B", "C"}, f"Missing letters in {lang}: {mc_counts}"
        assert len(set(mc_counts.values())) == 1, f"Uneven A/B/C in {lang}: {mc_counts}"


def test_midtrain_offline_pack_and_trainer_init_resume_smoke(tmp_path: Path):
    tok_path = ROOT / "tokenizers/kuon1-bilingual-24k.json"
    data_dir = tmp_path / "midtrain-smoke"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_kuon1_midtrain.py"),
            "--tokenizer", str(tok_path),
            "--out", str(data_dir),
            "--train-tokens", "40000",
            "--val-tokens", "3000",
            "--workers", "1",
            "--offline",
        ],
        cwd=str(ROOT),
        check=True,
    )
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizer_sha256"] == EXPECTED_TOKENIZER_SHA256
    assert meta["language_mix"] == {"en": 60, "id": 40}
    assert meta["tokens"]["en"]["blocked_eval_hits"] == 0
    assert meta["tokens"]["id"]["blocked_eval_hits"] == 0

    for split in ("train.en", "train.id", "val.en", "val.id"):
        arr = open_bin(data_dir, split)
        assert len(arr) > 500
        assert int(np.max(arr)) < meta["vocab_size"]

    # Smoke test 2-step continual pretrain (--init-from) + 2-step resume (--resume) on tiny model
    ckpt_dir = tmp_path / "ckpts"
    log_file = tmp_path / "log.jsonl"
    mcfg = ModelConfig(
        vocab_size=meta["vocab_size"], block_size=64, n_layer=4, n_head=2, n_kv_head=2,
        n_embd=32, ffn_hidden=64, loop_start=1, loop_end=3, loop_passes=2, loop_gate_rank=8,
    )
    tcfg = TrainConfig(
        data_dir=str(data_dir), checkpoint_dir=str(ckpt_dir), log_file=str(log_file),
        batch_size=2, grad_accum=5, steps=2, eval_interval=2, eval_batches=2,
        lr=6e-5, min_lr=5e-6, warmup_steps=1, seed=42,
    )
    res1 = train("kuon-midtrain-smoke", mcfg, tcfg, device="cpu", quiet=True)
    base_ckpt = Path(res1["ckpt"])
    assert base_ckpt.is_file()
    assert torch.load(base_ckpt, map_location="cpu", weights_only=False)["step"] == 2

    # Now --init-from base_ckpt for 2 steps, then --resume to step 4
    ckpt_dir2 = tmp_path / "ckpts2"
    tcfg2 = TrainConfig(
        data_dir=str(data_dir), checkpoint_dir=str(ckpt_dir2), log_file=str(tmp_path / "log2.jsonl"),
        batch_size=2, grad_accum=5, steps=4, eval_interval=2, eval_batches=2,
        lr=6e-5, min_lr=5e-6, warmup_steps=1, seed=42,
    )
    res2 = train("kuon-midtrain-smoke", mcfg, tcfg2, stop_step=2, init_from=str(base_ckpt), device="cpu", quiet=True)
    assert torch.load(res2["ckpt"], map_location="cpu", weights_only=False)["step"] == 2
    res3 = train("kuon-midtrain-smoke", mcfg, tcfg2, resume=res2["ckpt"], device="cpu", quiet=True)
    assert torch.load(res3["ckpt"], map_location="cpu", weights_only=False)["step"] == 4

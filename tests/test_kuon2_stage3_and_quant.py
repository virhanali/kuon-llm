"""Tests for upgraded Kuon-2 Stage 3 data symmetry/leakage and portable INT8 CPU quantization."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from kuon.config import ModelConfig
from kuon.model import Transformer
from kuon.quant import QUANT_FORMAT_INT8, dequantize_state_dict_int8, quantize_state_dict_int8
from kuon_agent.llm import KuonEngine
from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES
from scripts.eval_calibration import BUSINESS_CALIBRATION_CASES
from scripts.eval_confirmed_actions import CASES as CONFIRMED_CASES
from scripts.quantize_cpu import export_int8_checkpoint, verify_and_bench

ROOT = Path(__file__).resolve().parents[1]


def test_stage3_offline_sft_and_prefs_symmetry_and_zero_leakage(tmp_path: Path):
    tok = ROOT / "tokenizers/kuon1-bilingual-24k.json"
    sft_dir = tmp_path / "sft"
    prefs_dir = tmp_path / "prefs"

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_kuon1_sft.py"), "--tokenizer", str(tok), "--out", str(sft_dir), "--offline"],
        cwd=str(ROOT), check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_kuon1_prefs.py"), "--tokenizer", str(tok), "--out", str(prefs_dir), "--offline"],
        cwd=str(ROOT), check=True,
    )

    sft_train = [json.loads(l) for l in (sft_dir / "sft_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    prefs_train = [json.loads(l) for l in (prefs_dir / "prefs_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. Permutation-invariant A/B/C MC symmetry across BOTH business=False and business=True
    mc_sft = [r for r in sft_train if r["task"] == "confirmed_action_mc"]
    mc_counts = Counter(r["assistant"] for r in mc_sft)
    assert set(mc_counts.keys()) == {"A", "B", "C"}
    assert len(set(mc_counts.values())) == 1, f"skewed SFT MC letters: {mc_counts}"
    assert any(r["business"] is False for r in mc_sft) and any(r["business"] is True for r in mc_sft)

    mc_prefs = [r for r in prefs_train if r["task"] == "pref_confirmed_mc"]
    c_cnt = Counter(r["chosen"] for r in mc_prefs)
    r_cnt = Counter(r["rejected"] for r in mc_prefs)
    assert c_cnt == r_cnt and set(c_cnt.keys()) == {"A", "B", "C"}
    assert any(r["business"] is False for r in mc_prefs) and any(r["business"] is True for r in mc_prefs)

    # 2. New multi-metric lookup & entity-number binding hard negatives present
    assert any(r["task"] == "grounded_multi_metric" for r in sft_train)
    assert any(r["task"] == "biz_number_binding" for r in prefs_train)

    # 3. Strict zero leakage against every holdout benchmark prompt/issue
    forbidden_subs = []
    for c in CALIBRATION_CASES:
        forbidden_subs.append(c.user.casefold())
    for c in DELIVERABLE_CASES:
        forbidden_subs.append(c.user.casefold())
    for _, _, q, _, _ in BUSINESS_CALIBRATION_CASES:
        forbidden_subs.append(q.casefold())
    for scenarios in CONFIRMED_CASES.values():
        for issue, *_ in scenarios:
            forbidden_subs.append(issue.casefold())

    for row in sft_train + prefs_train:
        u = row.get("user", "").casefold()
        for bad in forbidden_subs:
            assert bad not in u, f"Eval leakage detected: {bad!r} in {u[:120]!r}"


def test_portable_int8_quantization_export_load_and_kuon_engine(tmp_path: Path):
    tok_src = ROOT / "tokenizers/kuon1-bilingual-24k.json"
    cfg = ModelConfig(
        vocab_size=24576, block_size=64, n_layer=4, n_embd=64,
        n_head=4, n_kv_head=2, ffn_hidden=128,
        loop_start=1, loop_end=3, loop_passes=2, loop_gate_rank=8,
    )
    torch.manual_seed(42)
    model = Transformer(cfg).eval()

    fp32_ckpt = tmp_path / "kuon-tiny-fp32.pt"
    int8_ckpt = tmp_path / "kuon-tiny-int8.pt"
    (tmp_path / "tokenizer.json").write_bytes(tok_src.read_bytes())
    torch.save({
        "model": model.state_dict(),
        "model_cfg": cfg.__dict__,
        "train_cfg": {"data_dir": str(tmp_path)},
        "step": 100,
        "val_loss": 2.5,
    }, fp32_ckpt)

    # Direct quantize/dequantize sanity check
    payload, stats = quantize_state_dict_int8(model.state_dict())
    assert stats["quantized_tensors"] > 10
    sd_rec = dequantize_state_dict_int8(payload, tie_word_embeddings=True)
    assert set(sd_rec.keys()) == set(model.state_dict().keys())

    # File export + fidelity + CPU decode benchmark
    exp = export_int8_checkpoint(fp32_ckpt, int8_ckpt)
    assert exp["dst_file_bytes"] < exp["src_file_bytes"] * 0.65
    loaded = torch.load(int8_ckpt, map_location="cpu", weights_only=False)
    assert loaded["quant"] == QUANT_FORMAT_INT8

    bench = verify_and_bench(fp32_ckpt, int8_ckpt, threads=2, decode_tokens=8)
    assert bench["mean_abs_logit_diff"] < 0.05
    assert bench["top1_agreement_pct"] >= 95.0
    assert bench["int8_decode_tok_s"] > 0

    # End-to-end KuonEngine CPU load & completion on quantized checkpoint
    eng = KuonEngine(str(int8_ckpt), device="cpu", temperature=0.0, quantize_cpu=True)
    reply = eng.complete("User: Halo!\nAssistant:", max_tokens=8)
    assert isinstance(reply, str)

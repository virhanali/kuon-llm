"""Tests for action-selection evaluation (randomized options + paired scoring)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_action_choice as eac


def test_build_items_randomizes_positions_and_preserves_gold():
    items = eac.build_items(eac.HOLDOUT_PAIRS, seed=42)
    assert len(items) == 6  # 2 scenarios * 3 cases
    gold_letters = {it["gold_letter"] for it in items}
    assert len(gold_letters) >= 2  # not all stuck on the same letter
    for it in items:
        assert it["letter_to_key"][it["gold_letter"]] == it["gold_key"]
        assert set(it["letter_to_key"].values()) == {"clarify", "action_a", "action_b"}


def test_pair_pass_requires_both_facts_correct():
    def rec(sc, case, mc_ok, mc_pred, cll_ok, cll_pred, ab_ok=None):
        return {
            "scenario": sc,
            "language": "en",
            "case": case,
            "mc_letter": {"pred_letter": "A", "pred_key": mc_pred, "correct": mc_ok},
            "completion_ll": {"pred_key": cll_pred, "correct": cll_ok, "ab_correct": ab_ok},
        }

    records = [
        # Scenario 1: both facts right -> pair_pass = 1
        rec("s1", "clarification", True, "clarify", True, "clarify"),
        rec("s1", "fact_a", True, "action_a", True, "action_a", ab_ok=True),
        rec("s1", "fact_b", True, "action_b", True, "action_b", ab_ok=True),
        # Scenario 2: only fact_a right -> pair_pass = 0
        rec("s2", "clarification", True, "clarify", True, "clarify"),
        rec("s2", "fact_a", True, "action_a", True, "action_a", ab_ok=True),
        rec("s2", "fact_b", False, "action_a", False, "action_a", ab_ok=False),
    ]
    mc_sum = eac.summarize_results(records, "mc_letter")
    assert mc_sum["pair_pass"] == "1/2"
    assert mc_sum["pair_changed_action"] == "1/2"
    assert mc_sum["full_3way_pass"] == "1/2"

    cll_sum = eac.summarize_results(records, "completion_ll")
    assert cll_sum["pair_pass"] == "1/2"
    assert cll_sum["ab_2way_pair_pass"] == "1/2"


def test_load_dev_scenarios(tmp_path):
    rows = [
        {"language": "id", "scenario": "id:demo", "task": "clarification",
         "user": "Omzet turun.", "assistant": "Yang turun pengunjung atau transaksi?"},
        {"language": "id", "scenario": "id:demo", "task": "adaptation",
         "user": "Omzet turun.\nAssistant: Yang turun pengunjung atau transaksi?\nUser: Pengunjung turun.",
         "assistant": "Perbaiki papan arah toko. Catat kunjungan harian."},
        {"language": "id", "scenario": "id:demo", "task": "adaptation",
         "user": "Omzet turun.\nAssistant: Yang turun pengunjung atau transaksi?\nUser: Transaksi turun.",
         "assistant": "Periksa stok barang terlaris. Catat transaksi harian."},
    ]
    p = tmp_path / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    scs = eac.load_dev_scenarios(p)
    assert len(scs) == 1
    assert scs[0]["action_a"] == "Perbaiki papan arah toko."
    assert scs[0]["action_b"] == "Periksa stok barang terlaris."

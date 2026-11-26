"""Tests for deterministic business analytical tools and continual-pretraining trainer init."""

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from kuon.config import ModelConfig, TrainConfig
from kuon.data import BinWriter
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer
from kuon.trainer import train
from kuon_agent.tools import FactLedger, business_kpi, safe_calc, summarize_table


def test_safe_calc_and_security():
    assert safe_calc("(120,000 - 78,000) / 120,000 * 100") == pytest.approx(35.0)
    assert safe_calc("-15 + 3.5 * 4") == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        safe_calc("__import__('os').system('echo nope')")
    with pytest.raises(ZeroDivisionError):
        safe_calc("10 / 0")


def test_business_kpis_exact():
    assert business_kpi("gross_margin", revenue=200, cogs=130)["margin_pct"] == 35.0
    assert business_kpi("pct_change", old=80, new=60)["pct_change"] == -25.0
    assert business_kpi("conversion_rate", buyers=45, visitors=300)["rate_pct"] == 15.0
    assert business_kpi("breakeven_units", fixed_cost=1000, unit_price=25, unit_cost=15)["units"] == 100
    assert business_kpi("eoq", annual_demand=1000, order_cost=50, holding_cost=2)["eoq_units"] == pytest.approx(223.61, abs=0.01)


def test_summarize_markdown_and_csv_tables():
    md = """
    | Week | Revenue | Orders |
    | --- | --- | --- |
    | W1 | $10,000 | 200 |
    | W2 | $8,500 | 170 |
    """
    s = summarize_table(md)
    assert s["row_count"] == 2
    assert s["numeric_summary"]["Revenue"]["sum"] == 18500.0
    assert s["numeric_summary"]["Revenue"]["first_to_last_pct"] == -15.0

    csv_txt = "Month,Margin\nJan,(500)\nFeb,1500\n"
    s2 = summarize_table(csv_txt)
    assert s2["numeric_summary"]["Margin"]["sum"] == 1000.0
    assert s2["numeric_summary"]["Margin"]["min"] == -500.0


def test_fact_ledger_dedup_and_citations():
    ledger = FactLedger()
    i1 = ledger.add("Road outside the bakery closed on Monday.", source="client")
    i2 = ledger.add("road outside the bakery closed on monday.", source="client")
    i3 = ledger.add("Gross margin is 35.0%.", source="tool:business_kpi")
    assert i1 == 1 and i2 == 1 and i3 == 2
    ev = ledger.format_evidence()
    assert "[1] (client) Road outside the bakery closed on Monday." in ev
    assert "[2] (tool:business_kpi) Gross margin is 35.0%." in ev


def test_trainer_init_from_continual_pretrain(tmp_path: Path):
    docs = ["toko kue jual roti manis pagi hari.", "sales margin increased by ten percent."] * 20
    tok = BPETokenizer.train(iter(docs), vocab_size=260, verbose=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tok.save(data_dir / "tokenizer.json")
    for split in ("train", "val"):
        w = BinWriter(data_dir / f"{split}.bin")
        for d in docs:
            w.write(tok.encode(d, add_eos=True))
        w.close()
    (data_dir / "meta.json").write_text(f'{{"vocab_size": {tok.vocab_size}}}')

    mcfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_embd=32,
                       n_head=2, n_kv_head=1, ffn_hidden=48, dropout=0.0)
    src = tmp_path / "stage2.pt"
    torch.save({
        "model": Transformer(mcfg).state_dict(),
        "model_cfg": asdict(mcfg),
        "train_cfg": {"data_dir": str(data_dir)},
        "step": 2500,
    }, src)

    tcfg = TrainConfig(
        data_dir=str(data_dir), steps=4, batch_size=2, grad_accum=1,
        eval_interval=2, eval_batches=2, lr=1e-3, min_lr=1e-4, warmup_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"), log_file=str(tmp_path / "log.jsonl"),
    )
    res = train("kuon-cpt-test", mcfg, tcfg, device="cpu", init_from=str(src), quiet=True)
    assert math.isfinite(res["best_val"])
    assert (tmp_path / "ckpts" / "kuon-cpt-test-last.pt").exists()


def test_business_agent_table_and_tool_call_loop():
    from kuon_agent.agent import AgentConfig, KuonAgent

    class ToolUsingStubEngine:
        name = "stub-business"

        def __init__(self):
            self.prompts: list[str] = []

        def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
            return len(prompt) + max_tokens * 4 < 4096

        def complete(self, prompt: str, max_tokens: int = 180) -> str:
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return "TOOL: kpi(gross_margin, revenue=10000, cogs=6500)"
            return "Gross margin is 35.0% [2]. Action: review top-3 cost items and track weekly margin."

    eng = ToolUsingStubEngine()
    agent = KuonAgent(eng, AgentConfig(use_rag=False, use_wiki=False, use_web=False))
    msg = "Here is our weekly table:\n| Week | Revenue | COGS |\n| --- | --- | --- |\n| W1 | 10000 | 6500 |\n| W2 | 8000 | 5600 |"
    res = agent.advise([("user", msg)])
    assert len(eng.prompts) == 2
    assert "tool:table_summary" in eng.prompts[0]
    assert "margin_pct=35.0" in eng.prompts[1]
    assert "35.0%" in res.answer
    assert any(s["title"] == "tool:table_summary" for s in res.sources)
    assert any(s["title"] == "tool:kpi" for s in res.sources)


def test_eval_business_agent_harness_and_leakage_disjoint():
    from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES, evaluate_engine
    from scripts.prepare_kuon1_sft import build_agentic_tool_rows

    tr, va = build_agentic_tool_rows()
    assert len(tr) >= 180 and len(va) >= 30
    train_users = {" ".join(r["user"].casefold().split()) for r in tr}
    for c in CALIBRATION_CASES:
        assert " ".join(c.user.casefold().split()) not in train_users
    for c in DELIVERABLE_CASES:
        assert " ".join(c.user.casefold().split()) not in train_users

    class HonestStubEngine:
        name = "honest-stub"

        def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
            return True

        def complete(self, prompt: str, max_tokens: int = 160) -> str:
            if "Gudang Surabaya" in prompt:
                return "Berdasarkan laporan [1], jumlah retur paket adalah 37 paket."
            if "Q2 Top-Line Sales Report" in prompt:
                return "The net profit margin is not specified in the report [1]. What were your total costs and taxes?"
            if "tool:table_summary" in prompt and "M1" in prompt:
                return "Fakta terkonfirmasi [1]: omzet turun -20% (total 90.000.000). Tindakan: uji sumber trafik dan ukur pesanan harian."
            return "Informasi tidak tersedia di bukti [1]. Mohon konfirmasi data yang kurang?"

    report = evaluate_engine(HonestStubEngine())
    assert report["pillar1_calibration"]["overall"]["total"] == 16
    assert report["pillar2_deliverables"]["overall"]["cases"] == 8
    assert report["pillar1_calibration"]["overall"]["unanswerable_restraint_pct"] == 100.0


def test_kpi_alias_normalization_and_abstain_regex():
    from scripts.eval_business_agent import ABSTAIN_RE
    from scripts.prepare_kuon1_prefs import build_counterfactual_business_prefs

    assert business_kpi("gross margin", revenue=200, cogs=130)["margin_pct"] == 35.0
    assert business_kpi("break-even_units", fixed_cost=1000, unit_price=25, unit_cost=15)["units"] == 100
    assert business_kpi("Average Order Value", revenue=500, orders=10)["aov"] == 50.0

    assert ABSTAIN_RE.search("The excerpts do not answer this question.")
    assert ABSTAIN_RE.search("Kutipan tersebut tidak menyebutkan jawaban untuk pertanyaan itu.")
    assert ABSTAIN_RE.search("I can’t confirm that from the documents provided.")

    tr, va = build_counterfactual_business_prefs()
    assert any(r["task"] == "pref_advice_vs_spurious_tool" for r in tr)
    assert any(r["task"] == "pref_advice_vs_spurious_tool" for r in va)


def test_trainer_stop_step_preserves_horizon(tmp_path: Path):
    docs = ["toko kue jual roti manis pagi hari.", "sales margin increased by ten percent."] * 20
    tok = BPETokenizer.train(iter(docs), vocab_size=260, verbose=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tok.save(data_dir / "tokenizer.json")
    for split in ("train", "val"):
        w = BinWriter(data_dir / f"{split}.bin")
        for d in docs:
            w.write(tok.encode(d, add_eos=True))
        w.close()
    (data_dir / "meta.json").write_text(f'{{"vocab_size": {tok.vocab_size}}}')

    mcfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_embd=32,
                       n_head=2, n_kv_head=1, ffn_hidden=48, dropout=0.0)
    tcfg = TrainConfig(
        data_dir=str(data_dir), steps=10, batch_size=2, grad_accum=1,
        eval_interval=5, eval_batches=2, lr=1e-3, min_lr=1e-4, warmup_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"), log_file=str(tmp_path / "log.jsonl"),
    )
    train("kuon-stop-test", mcfg, tcfg, device="cpu", steps=10, stop_step=3, quiet=True)
    ckpt = torch.load(tmp_path / "ckpts" / "kuon-stop-test-last.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 3


def test_trainer_resume_advances_data_rng(tmp_path: Path, monkeypatch):
    import kuon.trainer as trainer_mod

    docs = [f"toko nomor {i} menjual roti manis pagi hari di kota." for i in range(60)]
    tok = BPETokenizer.train(iter(docs), vocab_size=300, verbose=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tok.save(data_dir / "tokenizer.json")
    for split in ("train", "val"):
        w = BinWriter(data_dir / f"{split}.bin")
        for d in docs:
            w.write(tok.encode(d, add_eos=True))
        w.close()
    (data_dir / "meta.json").write_text(f'{{"vocab_size": {tok.vocab_size}}}')

    mcfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_embd=32,
                       n_head=2, n_kv_head=1, ffn_hidden=48, dropout=0.0)
    tcfg = TrainConfig(
        data_dir=str(data_dir), steps=10, batch_size=2, grad_accum=1,
        eval_interval=5, eval_batches=1, lr=1e-3, min_lr=1e-4, warmup_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"), log_file=str(tmp_path / "log.jsonl"),
    )

    seen_batches: list[list[int]] = []
    orig_get_batch = trainer_mod.get_batch

    def spy_get_batch(data, block_size, batch_size, device="cpu", *, rng=None):
        x, y = orig_get_batch(data, block_size, batch_size, device, rng=rng)
        seen_batches.append(x[0, :8].tolist())
        return x, y

    monkeypatch.setattr(trainer_mod, "get_batch", spy_get_batch)
    train("kuon-rng-test", mcfg, tcfg, device="cpu", steps=10, stop_step=2, quiet=True)
    first_run_batch0 = seen_batches[0]
    seen_batches.clear()
    train("kuon-rng-test", mcfg, tcfg, device="cpu", steps=10, stop_step=4,
          resume=str(tmp_path / "ckpts" / "kuon-rng-test-last.pt"), quiet=True)
    resumed_batch0 = seen_batches[0]
    assert resumed_batch0 != first_run_batch0


def test_sales_period_normalization_reordering_money_and_multiturn_followup():
    from kuon_agent.agent import AgentConfig, KuonAgent

    class DummyEngine:
        name = "dummy"
        def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
            return True
        def complete(self, prompt: str, max_tokens: int = 180) -> str:
            return "fallback"

    agent = KuonAgent(DummyEngine(), AgentConfig(use_rag=False, use_wiki=False, use_web=False))

    # 1. Natural period aliases ("Week 2 to Week 1" on W1/W2 table) -> +25%
    r1 = agent.advise([("user", "| Week | Revenue | Orders |\n|---|---|---|\n| W1 | 10000 | 100 |\n| W2 | 8000 | 70 |\nWhat is the revenue percentage change from Week 2 to Week 1?")])
    assert "W2 to W1: 25%" in r1.answer
    assert "W2=8000.0, W1=10000.0" in r1.notes

    # 2. Reordered rows ("Week 1 to Week 2" when W2 appears first) -> -20%
    r2 = agent.advise([("user", "| Week | Revenue | Orders |\n|---|---|---|\n| W2 | 8000 | 70 |\n| W1 | 10000 | 100 |\nWhat is the revenue percentage change from Week 1 to Week 2?")])
    assert "W1 to W2: -20%" in r2.answer

    # 3. Unknown explicit period -> refuses to guess row order
    r_ambig = agent.advise([("user", "| Week | Revenue |\n|---|---|\n| W1 | 10000 |\n| W2 | 8000 |\nWhat is the revenue percentage change from Week 3 to Week 1?")])
    assert "?" in r_ambig.answer and r_ambig.sources == []

    # 4. Indonesian Rupiah with dot-thousands -> -15,38%
    r_idr = agent.advise([("user", "| Minggu | Omzet |\n|---|---|\n| M1 | Rp9.750.000 |\n| M2 | Rp8.250.000 |\nBerapa persen perubahan omzet dari M1 ke M2?")])
    assert "-15,38%" in r_idr.answer
    assert "tool:kpi" in r_idr.sources[0]["title"]

    # 5. Mixed currencies -> refuses to compute silently
    r_mix = agent.advise([("user", "| Week | Revenue |\n|---|---|\n| W1 | Rp9.750.000 |\n| W2 | $8,250 |\nWhat is the revenue percentage change from W1 to W2?")])
    assert "different currencies" in r_mix.answer and r_mix.sources == []

    # 6. Indonesian natural paraphrase ("Hitung perubahan omzet dalam persen.") -> -20%
    r_para = agent.advise([("user", "| Minggu | Omzet |\n|---|---|\n| M1 | 10000 |\n| M2 | 8000 |\nHitung perubahan omzet dalam persen.")])
    assert "-20%" in r_para.answer and "M1 ke M2" in r_para.answer

    # 7. Multi-turn missing revenue -> follow-up fill -> recomputes -20%
    t1 = "| Week | Revenue |\n|---|---|\n| W1 | 10000 |\n| W2 | |\nWhat is the revenue percentage change from W1 to W2?"
    r_miss = agent.advise([("user", t1)])
    assert "W2" in r_miss.answer and "?" in r_miss.answer
    r_follow = agent.advise([
        ("user", t1),
        ("assistant", r_miss.answer),
        ("user", "W2 revenue was 8000."),
    ])
    assert "W1 to W2: -20%" in r_follow.answer
    assert "follow-up W2=8000" in r_follow.notes

    # 8. Multi-turn correction on existing cell
    r_corr = agent.advise([
        ("user", t1),
        ("assistant", r_miss.answer),
        ("user", "W2 revenue was 8000."),
        ("assistant", r_follow.answer),
        ("user", "Actually W2 revenue was 9000."),
    ])
    assert "W1 to W2: -10%" in r_corr.answer


def test_direct_kpi_and_tool_argument_provenance_guard():
    from kuon_agent.agent import AgentConfig, KuonAgent, _execute_tool_call

    class LyingToolEngine:
        name = "lying-stub"
        def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
            return True
        def complete(self, prompt: str, max_tokens: int = 180) -> str:
            # Emits fabricated COGS=40 when user wrote COGS=140
            return "TOOL: kpi(gross_margin, revenue=200, cogs=40)"

    agent = KuonAgent(LyingToolEngine(), AgentConfig(use_rag=False, use_wiki=False, use_web=False))

    # Direct grounded KPI handles explicit margin asks cleanly (both single-turn and multi-turn)
    r_id = agent.advise([("user", "Omzet bulan ini 17500000 dan HPP 11200000. Berapa margin kotor? Tolong pakai persentase.")])
    assert "36%" in r_id.answer and r_id.sources[0]["title"] == "tool:kpi"

    r_en = agent.advise([("user", "Revenue is 200 and COGS is 140. What is the gross margin?")])
    assert "Gross margin: 30%" in r_en.answer and "80%" not in r_en.answer

    # Multi-turn missing COGS -> follow-up HPP
    r_ask = agent.advise([("user", "Omzet bulan ini 17500000. Berapa margin kotor?")])
    assert "HPP" in r_ask.answer and "?" in r_ask.answer
    r_done = agent.advise([
        ("user", "Omzet bulan ini 17500000. Berapa margin kotor?"),
        ("assistant", r_ask.answer),
        ("user", "HPP 11200000."),
    ])
    assert "36%" in r_done.answer

    # Provenance guard blocks ungrounded model TOOL calls directly AND via agent loop
    out, err = _execute_tool_call("kpi", "gross_margin, revenue=200, cogs=40", allowed_numbers={200.0, 140.0})
    assert out is None and err and "ungrounded_args" in err

    # Even on a non-KPI-routed table prompt where model emits fabricated cogs=40, agent rejects it
    r_guard = agent.advise([("user", "| Item | Revenue | COGS |\n|---|---|---|\n| A | 200 | 140 |\n| B | 300 | 210 |\nSummarize our situation.")])
    assert "80%" not in r_guard.answer
    assert "tool_error(ungrounded_args)" in r_guard.notes
    assert not any(s["title"] == "tool:kpi" for s in r_guard.sources)


def test_strict_graders_and_disjoint_sft_prefs():
    from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES, score_calibration_answer, score_deliverable_answer
    from scripts.eval_calibration import score_aa_case
    from scripts.prepare_kuon1_sft import build_agentic_tool_rows

    # 1. Negated expected answer must NOT be graded correct
    id_ans_1 = next(c for c in CALIBRATION_CASES if c.case_id == "id_ans_1")
    assert score_calibration_answer(id_ans_1, "Berdasarkan laporan, bukan 37 paket.") == "hallucinate"

    # 2. Extra unsupported figure in advice must fail grounding (not 5/5)
    en_tbl = next(c for c in DELIVERABLE_CASES if c.case_id == "en_deliv_table_1")
    grade = score_deliverable_answer(
        en_tbl,
        "Revenue fell 25%. Compare order counts and average revenue per order. There were 731 site visits.",
        "",
    )
    assert grade["checks"]["no_unsupported_result_or_money"] is False
    assert grade["passed"] < grade["total"]

    # 3. Single-letter AA gold "B" must NOT match inside "bond"
    assert score_aa_case("It establishes a fixed-price bond, which is fixed.", "B") == (-1, "hallucinate")
    assert score_aa_case("The nominal face value of debt is denoted by B.", "B") == (1, "correct")

    # 4. SFT agentic tool rows: zero train/val missing-input overlap & diverse ROI outputs
    tr, va = build_agentic_tool_rows()
    tr_missing = {" ".join(r["user"].casefold().split()) for r in tr if r["task"] == "agent_tool_missing_input"}
    va_missing = {" ".join(r["user"].casefold().split()) for r in va if r["task"] == "agent_tool_missing_input"}
    assert tr_missing.isdisjoint(va_missing)
    roi_ev = {r["evidence"] for r in tr + va if r["task"] == "agent_tool_synthesis" and "kpi(roi)" in r.get("evidence", "")}
    roi_pcts = {ev.split("roi_pct=")[1].split(",")[0] for ev in roi_ev}
    assert len(roi_pcts) >= 6, f"expected diverse ROI percentages, got {roi_pcts}"


def test_repetition_controls_and_grounded_answer_verifier():
    from kuon.generate import _apply_repetition_controls
    from kuon_agent.tools import verify_grounded_answer

    # 1. N-gram blocking prevents repeating an already-emitted 3-gram
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    # Sequence already contains [0, 2, 1], ending with [0, 2] -> token 1 must be masked to -inf
    masked = _apply_repetition_controls(logits, [[0, 2, 1, 3, 0, 2]], repetition_penalty=1.15, no_repeat_ngram_size=3)
    assert masked[0, 1].item() == float("-inf")
    assert masked[0, 2].item() < logits[0, 2].item()

    # 2. Grounding verifier catches monetary-vs-headcount mismatch ("laba bersih ... 4.120 orang")
    ok, fb = verify_grounded_answer(
        "Berapa laba bersih cabang Palembang pada kuartal ketiga?",
        "[1] Laporan Kunjungan\nTotal pengunjung toko: 4.120 orang, dengan 980 transaksi selesai.",
        "Berdasarkan laporan kunjungan [1], laba bersih cabang Palembang Q3 adalah: 4.120 orang.",
        indonesian=True,
    )
    assert ok is False and fb and "belum menyebutkan" in fb

    # 3. Grounding verifier catches fabricated numbers not in evidence
    ok2, fb2 = verify_grounded_answer(
        "Berapa rata-rata waktu balas chat CS?",
        "[1] Rekap SLA\nShift malam: rata-rata balas 19 menit dengan 82 tiket masuk.",
        "Berdasarkan rekap SLA [1], jumlah tiket adalah 368 tiket masuk.",
        indonesian=True,
    )
    assert ok2 is False and fb2

    # 4. Valid grounded answer passes untouched
    ok3, fb3 = verify_grounded_answer(
        "Berapa rata-rata waktu balas chat CS pada shift malam?",
        "[1] Rekap SLA\nShift malam: rata-rata balas 19 menit dengan 82 tiket masuk.",
        "Berdasarkan rekap SLA [1], rata-rata balas shift malam adalah 19 menit.",
        indonesian=True,
    )
    assert ok3 is True and fb3 is None

    # 5. Currency mismatch guard catches Rp -> $ conversion hallucination
    ok_cur_bad, fb_cur_bad = verify_grounded_answer(
        "What is the Bandung branch net profit after tax?",
        "[1] keuangan_bandung.md\nLaba bersih cabang Bandung setelah pajak mencapai Rp42 juta.",
        "The Bandung branch net profit after tax was $42 million, as stated in [1].",
        indonesian=False,
    )
    assert ok_cur_bad is False and fb_cur_bad and "IDR" in fb_cur_bad

    # 6. Preserving original IDR/Rp currency passes cleanly
    ok_cur_good, fb_cur_good = verify_grounded_answer(
        "What is the Bandung branch net profit after tax?",
        "[1] keuangan_bandung.md\nLaba bersih cabang Bandung setelah pajak mencapai Rp42 juta.",
        "The Bandung branch net profit after tax was Rp42 million, as stated in [1].",
        indonesian=False,
    )
    assert ok_cur_good is True and fb_cur_good is None


def test_confirmed_mc_prefs_are_symmetric():
    from collections import Counter
    from scripts.prepare_kuon1_prefs import build_counterfactual_business_prefs

    tr, va = build_counterfactual_business_prefs()
    for split_name, rows in (("train", tr), ("val", va)):
        mc = [r for r in rows if r["task"] == "pref_confirmed_mc"]
        assert mc, f"{split_name} missing pref_confirmed_mc"
        if split_name == "train":
            assert any(r["business"] is False for r in mc) and any(r["business"] is True for r in mc)
        else:
            assert all(r["business"] is False for r in mc)
        chosen = Counter(r["chosen"] for r in mc)
        rejected = Counter(r["rejected"] for r in mc)
        assert set(chosen.keys()) == {"A", "B", "C"}, f"{split_name} chosen={chosen}"
        assert chosen == rejected, f"{split_name} asymmetric: chosen={chosen} rejected={rejected}"
        assert len(set(chosen.values())) == 1, f"{split_name} uneven across A/B/C: {chosen}"


def test_deterministic_table_analytics_id_en_tsv_and_currency_guard():
    from kuon_agent.agent import AgentConfig, KuonAgent

    class DummyEngine:
        name = "dummy"
        def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
            return True
        def complete(self, prompt: str, max_tokens: int = 180) -> str:
            return "fallback"

    agent = KuonAgent(DummyEngine(), AgentConfig(use_rag=False, use_wiki=False, use_web=False))

    tbl_id = (
        "Produk\tOmzet\tPesanan\n"
        "Teh Tarik\tRp41.000.000\t820\n"
        "Roti Panggang\tRp34.000.000\t680\n"
        "Kopi Hitam\tRp25.000.000\t500\n"
    )
    # Highest + share % from Excel/Sheets TSV paste
    r_max = agent.advise([("user", tbl_id + "Produk mana yang omzetnya terbesar?")])
    assert "Teh Tarik" in r_max.answer and "Rp41.000.000" in r_max.answer and "41%" in r_max.answer
    assert r_max.sources == [{"title": "tool:table_analytics", "url": ""}]

    # Multi-turn follow-up on the SAME pasted table: total & average pesanan
    r_sum = agent.advise([
        ("user", tbl_id + "Produk mana yang omzetnya terbesar?"),
        ("assistant", r_max.answer),
        ("user", "Berapa total pesanan semuanya?"),
    ])
    assert "2.000" in r_sum.answer

    # Lowest in English Markdown table
    tbl_en = "| Branch | Revenue | Orders |\n|---|---|---|\n| North | $12,000 | 120 |\n| East | $8,000 | 95 |\n| West | $20,000 | 210 |\n"
    r_min = agent.advise([("user", tbl_en + "Which branch had the lowest revenue?")])
    assert "East" in r_min.answer and "$8,000" in r_min.answer and "20%" in r_min.answer

    # Share breakdown
    r_share = agent.advise([("user", tbl_en + "What is each branch's revenue share?")])
    assert "West: 50%" in r_share.answer and "North: 30%" in r_share.answer and "East: 20%" in r_share.answer

    # Mixed currency guard
    tbl_mix = "| Branch | Revenue |\n|---|---|\n| A | Rp10.000.000 |\n| B | $5,000 |\nWhat is the total revenue?"
    r_mix = agent.advise([("user", tbl_mix)])
    assert "currencies" in r_mix.answer and r_mix.sources == []

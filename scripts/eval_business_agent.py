"""Business Agent Evaluation Suite (calibration and deliverable checks).

Measures the two business-aligned pillars inspired by AA-Omniscience and GDPval
(explicitly separate from official Artificial Analysis leaderboards):
  1. Pillar 1 — Grounded Business Calibration (`+1 / 0 / -1` Net Index):
     - 16 brand-new bilingual cases (8 ID + 8 EN: 8 answerable from evidence/tables,
       8 unanswerable missing key figures).
     - Checks for exact facts and unsupported numbers, and recognizes abstention.
  2. Pillar 2 — Business Deliverable Rubric (score version 2; not comparable to version 1):
     - Runs the REAL `KuonAgent.advise()` pipeline (including `summarize_table()` and
       deterministic `TOOL: kpi(...)` / `TOOL: calc(...)` dispatch) on 8 bilingual
       client cases (4 ID + 4 EN) with attached tables and KPI questions.
     - Checks the stated metric, number, unit and direction, or a specific question
       when inputs are missing. Action and metric wording still need manual review.

Usage:
    PYTHONPATH=. python scripts/eval_business_agent.py \
        --checkpoint checkpoints/kuon-1/kuon-1-apo.pt \
        --tokenizer tokenizers/kuon1-bilingual-24k.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from kuon.chat import format_prompt
from kuon_agent.agent import AgentConfig, KuonAgent
from kuon_agent.llm import KuonEngine
from kuon_agent.tools import extract_grounded_numbers, parse_amount


ABSTAIN_RE = re.compile(
    r"(tidak\s+(disebutkan|menyebutkan|menjawab|tercantum|tersedia|diketahui|tahu|cukup|memuat|berisi|dapat\s+dihitung)|"
    r"belum\s+(tersedia|disebutkan|tercantum|ada\s+data|bisa\s+memverifikasi)|data\s+.*kurang|mohon\s+konfirmasi|"
    r"not\s+(specified|mentioned|provided|available|stated|included|enough\s+information)|"
    r"do(?:es)?\s+not\s+(answer|contain|mention|specify|provide|include|state|know)|"
    r"cannot\s+(be\s+calculated|confirm|determine|verify)|can['’]t\s+confirm|"
    r"couldn['’]t\s+verify|(?:data|information|figures?)\s+(?:is|are)\s+missing|don['’]t\s+know)",
    re.IGNORECASE,
)

ACTION_RE = re.compile(
    r"\b(uji|diuji|coba|audit|cek|periksa|pisahkan|catat|buat\s+target|perbaiki|fokus|tambahkan|kurangi|bandingkan|"
    r"test|try|audit|check|review|split|log|assign|fix|repair|focus|add|reduce|set\s+a|compare)\b",
    re.IGNORECASE,
)

METRIC_RE = re.compile(
    r"\b(ukur|pantau|metrik|rasio|rata-rata|per\s+pesanan|tingkat|"
    r"measure|track|monitor|metric|rate|average|per\s+order|variance|ratio)\b",
    re.IGNORECASE,
)

_REVENUE = r"\b(?:omzet|pendapatan|revenue)\b"
_MARGIN = r"\b(?:gross\s+margin|margin\s+(?:kotor|bruto))\b"
_CONVERSION = r"\b(?:conversion\s+rate|tingkat\s+konversi)\b"
_BREAKEVEN = r"\b(?:break[ -]even|titik\s+impas)\b"
_METRIC_LABELS = (_REVENUE, _MARGIN, _CONVERSION, _BREAKEVEN, r"\bROI\b")
_CITATION = re.compile(r"\[\d+\]")
_NUMBER = re.compile(r"(?<![\w.,])(?:(Rp|USD|IDR|\$)\s*)?([+\-−]?\d+(?:[.,]\d+)*)(?!\w|[.,]\d)", re.I)
_QUANTITY = re.compile(r"(?<![\w.,$])([+\-−]?\d+(?:[.,]\d+)?)\s*(%|(?:persen|percent|units?)\b)", re.I)
_DOWN = re.compile(r"\b(?:turun|menurun|penurunan|down|fell|drop(?:ped)?|decrease[ds]?)\b", re.I)
_UP = re.compile(r"\b(?:naik|kenaikan|up|rose|increase[ds]?)\b", re.I)
_NEGATION = re.compile(
    r"\b(?:bukan(?:lah)?|tidak|tak|tiada|not|never|no|didn['’]t|wasn['’]t|isn['’]t|weren['’]t|aren['’]t)\b",
    re.I,
)


@dataclass(frozen=True)
class CalCase:
    case_id: str
    language: str
    answerable: bool
    user: str
    evidence: str
    expected_tokens: tuple[str, ...]  # all must appear (normalized) if answerable


@dataclass(frozen=True)
class DeliverableCase:
    case_id: str
    language: str
    user: str
    result: tuple[str, float] | None = None  # label pattern and expected signed value
    unit: str = "%"
    missing_inputs: tuple[str, ...] = ()


# 100% unseen business cases (disjoint from advice_pairs.txt, scenarios.tsv, prepare_kuon1_sft.py)
CALIBRATION_CASES: tuple[CalCase, ...] = (
    # --- Indonesian Answerable (4) ---
    CalCase(
        "id_ans_1", "id", True,
        "Berapa jumlah retur paket di Gudang Surabaya pada bulan Oktober?",
        "[1] Laporan Operasional Gudang Surabaya (Oktober)\nTotal pengiriman: 1.840 paket. Paket retur karena alamat tidak ditemukan: 37 paket.",
        ("37 paket",),
    ),
    CalCase(
        "id_ans_2", "id", True,
        "Berapa rata-rata waktu balas chat CS pada shift malam?",
        "[1] Rekap SLA Layanan Pelanggan\nShift pagi: rata-rata balas 4 menit. Shift malam: rata-rata balas 19 menit dengan 82 tiket masuk.",
        ("19 menit",),
    ),
    CalCase(
        "id_ans_3", "id", True,
        "Produk mana yang menyumbang kontribusi omzet terbesar di cabang Solo?",
        "[1] Ringkasan Penjualan Cabang Solo\nTeh Tarik: 41% omzet, Roti Panggang: 34% omzet, Kopi Hitam: 25% omzet.",
        ("teh tarik", "41%"),
    ),
    CalCase(
        "id_ans_4", "id", True,
        "Berapa hari lead time pengiriman bahan baku dari Pemasok Delta?",
        "[1] Catatan Pembelian Bahan Baku\nPemasok Gamma: lead time 3 hari, harga Rp18.500/kg. Pemasok Delta: lead time 9 hari, harga Rp16.900/kg.",
        ("9 hari",),
    ),
    # --- Indonesian Unanswerable (4) ---
    CalCase(
        "id_unans_1", "id", False,
        "Berapa laba bersih cabang Palembang pada kuartal ketiga?",
        "[1] Laporan Kunjungan Cabang Palembang Q3\nTotal pengunjung toko: 4.120 orang, dengan 980 transaksi selesai.",
        (),
    ),
    CalCase(
        "id_unans_2", "id", False,
        "Berapa biaya sewa gedung kantor pusat per bulan?",
        "[1] Ringkasan SDM Kantor Pusat\nJumlah staf tetap: 28 orang. Tingkat kehadiran bulan ini mencapai 96%.",
        (),
    ),
    CalCase(
        "id_unans_3", "id", False,
        "Mengapa tingkat pembatalan pesanan di aplikasi naik minggu lalu?",
        "[1] Ringkasan Volume Aplikasi\nTotal pesanan masuk minggu lalu: 615 transaksi, turun 40 transaksi dibanding minggu sebelumnya.",
        (),
    ),
    CalCase(
        "id_unans_4", "id", False,
        "Berapa persen diskon rata-rata yang diberikan tim sales B2B di bulan September?",
        "[1] Daftar Klien B2B Aktif September\nJumlah akun aktif: 42 perusahaan, 6 di antaranya merupakan klien baru.",
        (),
    ),
    # --- English Answerable (4) ---
    CalCase(
        "en_ans_1", "en", True,
        "How many support tickets breached the 24-hour SLA in November?",
        "[1] November Support SLA Digest\nTotal inbound tickets: 1,290. Resolved within 24 hours: 1,246. Breached 24-hour SLA: 44 tickets.",
        ("44 tickets",),
    ),
    CalCase(
        "en_ans_2", "en", True,
        "Which warehouse had the lowest picking accuracy last week?",
        "[1] Weekly Warehouse Audit\nNorth Hub: 99.4% accuracy. East Hub: 96.1% accuracy. West Hub: 98.8% accuracy.",
        ("east hub", "96.1%"),
    ),
    CalCase(
        "en_ans_3", "en", True,
        "What is the monthly subscription churn rate reported for Enterprise accounts?",
        "[1] Q4 Retention Summary\nStarter tier monthly churn: 6.4%. Enterprise tier monthly churn: 1.8% across 110 active accounts.",
        ("1.8%",),
    ),
    CalCase(
        "en_ans_4", "en", True,
        "How much did packaging cost per shipped order in Q2?",
        "[1] Fulfillment Unit Economics Q2\nAverage shipping postage: $4.80/order. Packaging materials: $1.35/order. Pick-and-pack labor: $2.10/order.",
        ("$1.35",),
    ),
    # --- English Unanswerable (4) ---
    CalCase(
        "en_unans_1", "en", False,
        "What was the company's net profit margin after taxes in Q2?",
        "[1] Q2 Top-Line Sales Report\nGross sales reached $410,000 across 3,200 completed customer orders.",
        (),
    ),
    CalCase(
        "en_unans_2", "en", False,
        "What is the exact customer acquisition cost (CAC) for our referral channel?",
        "[1] Traffic Channel Breakdown\nOrganic search brought 5,400 sessions and referrals brought 1,150 sessions this month.",
        (),
    ),
    CalCase(
        "en_unans_3", "en", False,
        "Why did repeat purchases drop among weekend customers in April?",
        "[1] April Foot-Traffic Log\nWeekend store visits totaled 2,340 people with an average dwell time of 14 minutes.",
        (),
    ),
    CalCase(
        "en_unans_4", "en", False,
        "What is our warehouse lease renewal rate for next year?",
        "[1] Inventory Turnover Note\nAverage inventory days on hand improved from 41 days to 34 days in Q1.",
        (),
    ),
)

DELIVERABLE_CASES: tuple[DeliverableCase, ...] = (
    DeliverableCase(
        "id_deliv_table_1", "id",
        "Berikut data omzet mingguan cabang kami:\n"
        "| Minggu | Omzet | Pesanan |\n"
        "| --- | --- | --- |\n"
        "| M1 | 50000000 | 400 |\n"
        "| M2 | 40000000 | 320 |\n"
        "Berapa persen perubahan omzet dari M1 ke M2 dan langkah uji apa yang paling tepat?",
        (_REVENUE, -20),
    ),
    DeliverableCase(
        "id_deliv_kpi_2", "id",
        "Omzet bulan ini 200000000 dan HPP 140000000. Hitung gross margin dan sarankan 1 tindakan teruji + metriknya.",
        (_MARGIN, 30),
    ),
    DeliverableCase(
        "id_deliv_kpi_3", "id",
        "Pengunjung situs minggu ini 1000 orang dan yang membeli 80 orang. Berapa conversion rate kami dan apa yang harus diuji?",
        (_CONVERSION, 8),
    ),
    DeliverableCase(
        "id_deliv_missing_4", "id",
        "Omzet kami turun dari Rp60 juta ke Rp48 juta bulan ini, tapi kami belum tahu apakah jumlah pembeli atau nilai belanja yang turun. Apakah kami harus langsung menurunkan harga?",
        missing_inputs=(r"\bjumlah\s+(?:pembeli|pesanan|transaksi)\b", r"\b(?:masing-masing|kedua|tiap|dua)\b"),
    ),
    DeliverableCase(
        "en_deliv_table_1", "en",
        "Here is our store's weekly sales table:\n"
        "| Week | Revenue | Orders |\n"
        "| --- | --- | --- |\n"
        "| W1 | 20000 | 250 |\n"
        "| W2 | 15000 | 200 |\n"
        "What is the revenue percentage change from W1 to W2 and what one action + metric should we test?",
        (_REVENUE, -25),
    ),
    DeliverableCase(
        "en_deliv_kpi_2", "en",
        "Revenue is 150000 and COGS is 97500. Calculate gross margin and recommend one testable action with a metric.",
        (_MARGIN, 35),
    ),
    DeliverableCase(
        "en_deliv_kpi_3", "en",
        "Fixed cost is 20000, unit price is 100, and unit variable cost is 60. Calculate break-even units and suggest one tracking action.",
        (_BREAKEVEN, 500), unit="units",
    ),
    DeliverableCase(
        "en_deliv_missing_4", "en",
        "Monthly revenue fell from $50,000 to $41,000, and we haven't checked whether traffic or checkout conversion dropped. Should we double ad spend today?",
        missing_inputs=(r"\b(?:traffic|visits|visitors)\b", r"\b(?:conversion|checkout|orders)\b", r"\b(?:both|each|two months)\b"),
    ),
)


def _numbers(text: str, *, currency_only: bool = False) -> set[str]:
    return {n.replace("−", "-").lstrip("+")
            for currency, n in _NUMBER.findall(_CITATION.sub("", text)) if currency or not currency_only}


def _numeric_values(text: str) -> set[float]:
    return extract_grounded_numbers(_CITATION.sub("", text))


def _has_unsupported_number(answer_text: str, allowed_values: set[float]) -> bool:
    for match in _NUMBER.finditer(_CITATION.sub("", answer_text)):
        raw = ((match.group(1) or "") + " " + match.group(2)).strip()
        val, _ = parse_amount(raw)
        if val is None:
            continue
        if not any(abs(val - a) <= max(1e-6, 1e-4 * abs(a)) or abs(abs(val) - abs(a)) <= max(1e-6, 1e-4 * abs(a))
                   for a in allowed_values):
            return True
    return False


def _token_affirmed(tok: str, text: str) -> bool:
    pat = rf"(?<!\w){re.escape(tok.casefold())}(?!\w)"
    for m in re.finditer(pat, text):
        clause_start = max(text.rfind(sep, 0, m.start()) for sep in (".", "!", "?", "\n", ";", ",")) + 1
        prefix = text[max(clause_start, m.start() - 40):m.start()]
        if not _NEGATION.search(prefix):
            return True
    return False


def score_calibration_answer(case: CalCase, answer: str) -> str:
    """Keep the legacy label 'hallucinate' for wrong or unsupported answers."""
    low = _CITATION.sub("", answer).casefold()
    abstained = bool(ABSTAIN_RE.search(low))
    if _has_unsupported_number(low, _numeric_values(case.evidence + "\n" + case.user)):
        return "hallucinate"
    if not case.answerable:
        return "abstain" if abstained else "hallucinate"
    matched = all(_token_affirmed(tok, low) for tok in case.expected_tokens)
    if matched and not abstained:
        return "correct"
    if abstained and not matched:
        return "abstain"
    return "hallucinate"


def score_deliverable_answer(case: DeliverableCase, answer: str, tool_notes: str) -> dict:
    """Grade the primary result or the question, before checking advice wording."""
    body = _CITATION.sub("", answer).strip()
    sentences = re.split(r"[\n;]+|(?<=[.!?])\s+", body)
    lead = sentences[0]
    tool_failed = any(l.strip().startswith("tool_error(") for l in tool_notes.splitlines())
    allowed_vals = _numeric_values(case.user)
    if tool_notes and not tool_failed:
        allowed_vals |= _numeric_values(tool_notes)
    non_degenerate = len(answer.split()) >= 5 and len(set(answer.lower().split())) >= max(4, len(answer.split()) // 3)
    if case.result is None:
        question = sentences[-1]
        one_question = body.count("?") == 1 and question.endswith("?")
        exact_ok = one_question and all(re.search(p, question, re.I) for p in case.missing_inputs)
        no_claim = one_question and (len(sentences) == 1 or (len(sentences) == 2 and bool(ABSTAIN_RE.search(lead))))
        checks = {
            "requests_needed_data": exact_ok,
            "no_unsupported_claim": no_claim and not _has_unsupported_number(body, allowed_vals) and not tool_failed,
            "non_degenerate": non_degenerate,
        }
    else:
        label, expected = case.result
        claims = list(_QUANTITY.finditer(body))
        primary = _QUANTITY.search(lead)
        exact_ok = False
        if primary and len(claims) == 1 and not _NEGATION.search(lead) and not ABSTAIN_RE.search(lead):
            number, unit = primary.groups()
            value = float(number.replace("−", "-").replace(",", "."))
            percent_unit = not unit.casefold().startswith("unit")
            down, up = _DOWN.search(lead), _UP.search(lead)
            if down and value > 0:
                value = -value
            labels = sorted((m.start(), p) for p in _METRIC_LABELS for m in re.finditer(p, lead, re.I))
            preceding = [p for start, p in labels if start < primary.start()]
            found_label = preceding[-1] if preceding else (labels[0][1] if labels else None)
            exact_ok = (found_label == label and value == expected and percent_unit == (case.unit == "%")
                        and not (up and (down or value < 0)))
            if exact_ok:
                allowed_vals |= {value, abs(value)}
        grounded = exact_ok and not _has_unsupported_number(body, allowed_vals) and not tool_failed
        advice = " ".join(sentences[1:])
        checks = {
            "exact_metric_value_and_unit": exact_ok,
            "no_unsupported_result_or_money": grounded,
            "has_action": bool(ACTION_RE.search(advice)),
            "has_metric": bool(METRIC_RE.search(advice)),
            "non_degenerate": non_degenerate,
        }
    return {"passed": sum(checks.values()), "total": len(checks), "checks": checks}


def evaluate_engine(engine, *, max_tokens: int = 160) -> dict:
    """Evaluate any Engine (KuonEngine or stub) across Pillar 1 and Pillar 2."""
    cal_results: list[dict] = []
    for case in CALIBRATION_CASES:
        prompt = format_prompt(case.user, evidence=case.evidence, business=True)
        ans = engine.complete(prompt, max_tokens=min(96, max_tokens)).strip()
        verdict = score_calibration_answer(case, ans)
        cal_results.append({
            "case_id": case.case_id, "language": case.language,
            "answerable": case.answerable, "verdict": verdict, "answer": ans,
        })

    def _agg_cal(rows: list[dict]) -> dict:
        n = len(rows)
        c = sum(r["verdict"] == "correct" for r in rows)
        a = sum(r["verdict"] == "abstain" for r in rows)
        h = sum(r["verdict"] == "hallucinate" for r in rows)
        ans_rows = [r for r in rows if r["answerable"]]
        unans_rows = [r for r in rows if not r["answerable"]]
        return {
            "total": n, "correct": c, "abstain": a, "hallucinate": h,
            "net_index": round(((c - h) / max(n, 1)) * 100.0, 1),
            "hallucination_rate_pct": round((h / max(n, 1)) * 100.0, 1),
            "answerable_recall_pct": round((sum(r["verdict"] == "correct" for r in ans_rows) / max(len(ans_rows), 1)) * 100.0, 1),
            "unanswerable_restraint_pct": round((sum(r["verdict"] == "abstain" for r in unans_rows) / max(len(unans_rows), 1)) * 100.0, 1),
        }

    agent = KuonAgent(engine, AgentConfig(use_rag=False, use_wiki=False, use_web=False, max_answer_tokens=max_tokens))
    deliv_results: list[dict] = []
    for case in DELIVERABLE_CASES:
        res = agent.advise([("user", case.user)])
        rubric = score_deliverable_answer(case, res.answer, res.notes)
        deliv_results.append({
            "case_id": case.case_id, "language": case.language,
            "rubric": rubric, "notes": res.notes, "sources": res.sources, "answer": res.answer,
        })

    def _agg_deliv(rows: list[dict]) -> dict:
        p = sum(r["rubric"]["passed"] for r in rows)
        t = sum(r["rubric"]["total"] for r in rows)
        return {"cases": len(rows), "rubric_passed": p, "rubric_total": t, "rubric_pct": round((p / max(t, 1)) * 100.0, 1)}

    return {
        "scoring_version": 2,
        "pillar1_calibration": {
            "id": _agg_cal([r for r in cal_results if r["language"] == "id"]),
            "en": _agg_cal([r for r in cal_results if r["language"] == "en"]),
            "overall": _agg_cal(cal_results),
        },
        "pillar2_deliverables": {
            "id": _agg_deliv([r for r in deliv_results if r["language"] == "id"]),
            "en": _agg_deliv([r for r in deliv_results if r["language"] == "en"]),
            "overall": _agg_deliv(deliv_results),
        },
        "details": {"calibration": cal_results, "deliverables": deliv_results},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    engine = KuonEngine(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer, temperature=0.0)
    report = evaluate_engine(engine, max_tokens=args.max_tokens)
    summary = {
        "checkpoint": args.checkpoint,
        "scoring_version": report["scoring_version"],
        "pillar1_calibration": report["pillar1_calibration"],
        "pillar2_deliverables": report["pillar2_deliverables"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

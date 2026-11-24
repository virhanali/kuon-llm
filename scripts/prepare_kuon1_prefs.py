"""Build Stage 3B Context-APO-zero / ORPO Preference Pairs for Kuon 0.1.

Grounded in Context-DPO (arXiv:2412.15280), Abstain-R1 (ACL Findings 2026), and
Anchored Preference Optimization (TACL 2025 / SmolLM3), every pair targets an observed
failure mode from Stage 3 v1:
  1. Counterfactual Fact-Flip Hard Negatives:
       (opening + fact_a) -> chosen=answer_a, rejected=answer_b (and vice versa).
  2. Cross-Scenario Mode-Collapse Hard Negatives:
       (opening + fact_a) -> chosen=answer_a, rejected=unrelated memorized scenario answer
       (directly punishes outputting "smaller packaging" for a road-closure problem).
  3. Missing-Fact Clarification vs Premature Guessing:
       (opening without key fact) -> chosen=clarifying_question, rejected=premature advice.
  4. Anti-Over-Abstention & Context-Faithfulness Pairs (EN + ID):
       - Answerable grounded QA -> chosen=grounded answer, rejected="I do not know." / wrong span.
       - Unanswerable grounded QA -> chosen=honest abstention + clarification, rejected=fabricated claim.
  5. General Quality Anchor:
       - Clean high-margin pairs from HuggingFaceH4/ultrafeedback_binarized (MIT).

Strictly evaluation-disjoint: advice_dev (6 EN + 6 ID) goes ONLY to prefs_val.jsonl,
and advice_eval.jsonl / eval.jsonl are excluded completely.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from kuon.chat import build_example
from kuon.tokenizer import BPETokenizer
from scripts.prepare_kuon1_corpus import HttpRangeFile, USER_AGENT
from scripts.prepare_kuon1_sft import (
    IRRELEVANT_NOTES,
    build_agentic_tool_rows,
    build_balanced_calibration_rows,
    build_grounded_business_rows,
    build_squad_v2_rows,
    build_tydiqa_indonesian_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _resolve_url(url: str) -> str:
    req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as r:
        return r.geturl()


def build_counterfactual_business_prefs() -> tuple[list[dict], list[dict]]:
    """Build fact-flip, cross-scenario, invariance, and clarification preference pairs."""
    biz_root = ROOT / "datasets/business"
    groups: dict[str, list[tuple[str, str, str, str, str, str, str]]] = defaultdict(list)
    for line in (biz_root / "advice_pairs.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        lang, name, opening, question, fact_a, answer_a, fact_b, answer_b = line.split("|")
        groups[lang].append((name, opening, question, fact_a, answer_a, fact_b, answer_b))

    rng = random.Random(1337)
    train_pairs: list[dict] = []
    val_pairs: list[dict] = []

    for lang in ("id", "en"):
        cases = list(groups[lang])
        assert len(cases) == 30
        rng.shuffle(cases)
        for is_val, subset in ((False, cases[:24]), (True, cases[24:])):
            target = val_pairs if is_val else train_pairs
            n = len(subset)
            notes = IRRELEVANT_NOTES[lang]
            for idx, (name, opening, question, fact_a, answer_a, fact_b, answer_b) in enumerate(subset):
                meta = {"language": lang, "scenario": f"{lang}:{name}", "business": True}
                # Pick a deterministic unrelated scenario from the SAME split (zero leakage!)
                other_a = subset[(idx + 3) % n][4]
                other_b = subset[(idx + 5) % n][6]
                other_q = subset[(idx + 4) % n][2]

                # 1. Missing-fact clarification chosen over premature action or wrong question
                target.append({**meta, "task": "pref_clarify_vs_guess", "user": opening, "chosen": question, "rejected": answer_a})
                target.append({**meta, "task": "pref_clarify_vs_guess", "user": opening, "chosen": question, "rejected": answer_b})
                if not is_val:
                    target.append({**meta, "task": "pref_clarify_vs_unrelated", "user": opening, "chosen": question, "rejected": other_q})

                # 2. Counterfactual fact flips (direct, multiturn, evidence, invariance)
                for f_idx, (fact, good, bad, cross_bad) in enumerate((
                    (fact_a, answer_a, answer_b, other_a),
                    (fact_b, answer_b, answer_a, other_b),
                )):
                    note = notes[(idx + f_idx) % len(notes)]
                    # Multiturn fact flip
                    target.append({
                        **meta, "task": "pref_fact_flip_multiturn",
                        "user": f"{opening}\nAssistant: {question}\nUser: {fact}",
                        "chosen": good, "rejected": bad,
                    })
                    # Direct fact flip
                    target.append({
                        **meta, "task": "pref_fact_flip_direct",
                        "user": f"{opening} {fact}",
                        "chosen": good, "rejected": bad,
                    })
                    # Cross-scenario mode-collapse negative (punishes spitting unrelated memorized advice!)
                    target.append({
                        **meta, "task": "pref_cross_scenario",
                        "user": f"{opening}\nAssistant: {question}\nUser: {fact}",
                        "chosen": good, "rejected": cross_bad,
                    })
                    # Anti-spurious-tool-call negative (prevents emitting TOOL: kpi on non-numeric operational advice)
                    if (idx + f_idx) % 2 == 0:
                        bogus_tool = (
                            "TOOL: kpi(conversion_rate, buyers=100, visitors=500)"
                            if f_idx == 0
                            else "TOOL: kpi(gross_margin, revenue=50000, cogs=30000)"
                        )
                        target.append({
                            **meta, "task": "pref_advice_vs_spurious_tool",
                            "user": f"{opening} {fact}",
                            "chosen": good, "rejected": bogus_tool,
                        })
                    if not is_val:
                        # Evidence block fact flip
                        target.append({
                            **meta, "task": "pref_fact_flip_evidence",
                            "user": opening,
                            "evidence": f"[1] {fact}\n[2] {note}",
                            "chosen": good, "rejected": bad,
                        })
                        # Cross-scenario on direct prompt
                        target.append({
                            **meta, "task": "pref_cross_scenario_direct",
                            "user": f"{opening} {fact}",
                            "chosen": good, "rejected": cross_bad,
                        })
                        # Irrelevant detail invariance
                        target.append({
                            **meta, "task": "pref_invariance",
                            "user": f"{opening} {fact} {note}",
                            "chosen": good, "rejected": bad,
                        })

                # 3. Permutation-invariant multiple-choice A/B/C preference pairs (Confirmed facts format)
                hdr = "Fakta terkonfirmasi" if lang == "id" else "Confirmed facts"
                instr = (
                    "Pilih satu tindakan dan ukuran yang cocok. Jika fakta kurang, pilih pertanyaan."
                    if lang == "id"
                    else "Choose one action with its metric. If a fact is missing, choose the question."
                )
                missing_fact = (
                    "Belum diketahui di bagian mana proses atau pelanggan tertahan."
                    if lang == "id"
                    else "It is not yet known where customers or operations are getting held up."
                )
                date_note = (
                    ("Catatan dicatat hari Selasa." if idx % 2 == 0 else "Catatan dicatat hari Rabu.")
                    if lang == "id"
                    else ("The note was logged on Tuesday." if idx % 2 == 0 else "The note was logged on Wednesday.")
                )
                choices = (answer_a, answer_b, question)
                # Symmetric, balanced A/B/C preference pairs (business=False to match eval_confirmed_actions):
                # Every gold letter is paired against BOTH wrong letters so chosen & rejected counts are
                # strictly 1:1:1 across A, B, C (zero letter bias) and stay ~4.5% of total pref pairs.
                if is_val or idx % 2 == 0:
                    for fact_i, (confirmed, exp_idx) in enumerate(((fact_a, 0), (fact_b, 1), (missing_fact, 2))):
                        body = f"{opening}\n{hdr}: {confirmed} {date_note}\n{instr}"
                        offset = (idx + 2 * fact_i) % 3
                        order = [(offset + k) % 3 for k in range(3)]
                        opts_txt = "\n".join(f"{ltr}. {choices[c_i]}" for ltr, c_i in zip("ABC", order))
                        gold_letter = "ABC"[order.index(exp_idx)]
                        for wrong_letter in (l for l in "ABC" if l != gold_letter):
                            target.append({
                                **meta, "task": "pref_confirmed_mc",
                                "business": False,
                                "user": f"{body}\n{opts_txt}",
                                "chosen": gold_letter,
                                "rejected": wrong_letter,
                            })
    # Guard against asymmetric A/B/C letter bias
    for split_pairs in (train_pairs, val_pairs):
        mc_rows = [r for r in split_pairs if r["task"] == "pref_confirmed_mc"]
        if mc_rows:
            c_cnt = Counter(r["chosen"] for r in mc_rows)
            r_cnt = Counter(r["rejected"] for r in mc_rows)
            assert c_cnt == r_cnt and set(c_cnt.keys()) == {"A", "B", "C"}, (
                f"Asymmetric MC preference distribution: chosen={dict(c_cnt)} rejected={dict(r_cnt)}"
            )
    return train_pairs, val_pairs


def build_calibration_prefs(offline: bool = False) -> tuple[list[dict], list[dict]]:
    """Build anti-over-abstention + anti-hallucination pairs (Context-DPO + Abstain-R1)."""
    train_pairs: list[dict] = []
    val_pairs: list[dict] = []

    def pair_grounded(rows: list[dict], out: list[dict], tag: str):
        ans_rows = [
            r for r in rows
            if "unans" not in r["task"]
            and r["task"] not in ("abstain", "rewrite", "summarize", "decision_missing_drivers")
        ]
        unans_rows = [r for r in rows if "unans" in r["task"] or r["task"] == "abstain"]
        decision_rows = [r for r in rows if r["task"] == "decision_missing_drivers"]

        # Answerable -> punish false abstention AND wrong-entity / wrong-passage hallucination
        for i, r in enumerate(ans_rows):
            refuse = "Saya tidak tahu." if r["language"] == "id" else "I do not know."
            out.append({
                "language": r["language"], "task": f"{tag}_anti_false_refusal",
                "business": r.get("business", True), "user": r["user"], "evidence": r.get("evidence"),
                "chosen": r["assistant"], "rejected": refuse,
            })
            if r.get("distractor"):
                # Hard counterfactual negative: wrong entity / first-number shortcut from the SAME passage
                out.append({
                    "language": r["language"], "task": f"{tag}_entity_faithful",
                    "business": r.get("business", True), "user": r["user"], "evidence": r.get("evidence"),
                    "chosen": r["assistant"], "rejected": r["distractor"],
                })
            elif len(ans_rows) > 2:
                wrong = ans_rows[(i + 1) % len(ans_rows)]["assistant"]
                if wrong != r["assistant"]:
                    out.append({
                        "language": r["language"], "task": f"{tag}_context_faithful",
                        "business": r.get("business", True), "user": r["user"], "evidence": r.get("evidence"),
                        "chosen": r["assistant"], "rejected": wrong,
                    })

        # Unanswerable -> punish copying a distractor number from the passage or guessing
        for i, r in enumerate(unans_rows):
            if r.get("distractor"):
                guess = r["distractor"]
            else:
                guess = (
                    "Berdasarkan perkiraan umum, angkanya sekitar 25% atau Rp15 juta minggu ini."
                    if r["language"] == "id"
                    else "Based on typical figures, the exact result is 24.5% for this period."
                )
                if ans_rows:
                    cand = ans_rows[i % len(ans_rows)]["assistant"]
                    if cand != r["assistant"]:
                        guess = cand
            out.append({
                "language": r["language"], "task": f"{tag}_anti_hallucination",
                "business": r.get("business", True), "user": r["user"], "evidence": r.get("evidence"),
                "chosen": r["assistant"], "rejected": guess,
            })

        # Numeric decision questions with unknown drivers -> punish spurious TOOL: kpi(),
        # premature actions, AND vague single-period / wrong-metric questions
        for r in decision_rows:
            spurious_tool = (
                "TOOL: kpi(pct_change, old=80000000, new=64000000)"
                if r["language"] == "id"
                else "TOOL: kpi(pct_change, old=50000, new=41000)"
            )
            premature = (
                "Ya, langsung turunkan harga 20% minggu ini agar omzet segera naik."
                if r["language"] == "id"
                else "Yes, double your ad spend immediately today to recover revenue."
            )
            vague_q = (
                "Berapa omzet yang turun pada bulan tersebut?"
                if r["language"] == "id"
                else "What was your total revenue in that week?"
            )
            out.append({
                "language": r["language"], "task": "pref_decision_vs_spurious_tool",
                "business": True, "user": r["user"], "chosen": r["assistant"], "rejected": spurious_tool,
            })
            out.append({
                "language": r["language"], "task": "pref_decision_vs_premature",
                "business": True, "user": r["user"], "chosen": r["assistant"], "rejected": premature,
            })
            out.append({
                "language": r["language"], "task": "pref_decision_vs_vague",
                "business": True, "user": r["user"], "chosen": r["assistant"], "rejected": vague_q,
            })

    gb_tr, gb_va = build_grounded_business_rows()
    pair_grounded(gb_tr, train_pairs, "biz")
    pair_grounded(gb_va, val_pairs, "biz")

    cal_tr, cal_va = build_balanced_calibration_rows()
    for src, dst in ((cal_tr, train_pairs), (cal_va, val_pairs)):
        for r in src:
            if r["task"] == "calibration_known_concept":
                dst.append({
                    "language": "en", "task": "cal_known_vs_abstain", "business": False,
                    "user": r["user"], "chosen": r["assistant"], "rejected": "I do not know.",
                })
            else:
                dst.append({
                    "language": "en", "task": "cal_unknown_vs_guess", "business": False,
                    "user": r["user"], "chosen": r["assistant"],
                    "rejected": "The exact reported figure was 14.8 million based on the regulatory filing.",
                })

    if not offline:
        sq_tr, sq_va = build_squad_v2_rows(ans_limit=160, unans_limit=100)
        pair_grounded(sq_tr, train_pairs, "squad")
        pair_grounded(sq_va, val_pairs, "squad")

        ty_tr, ty_va = build_tydiqa_indonesian_rows(ans_limit=160, unans_limit=100)
        pair_grounded(ty_tr, train_pairs, "tydi_id")
        pair_grounded(ty_va, val_pairs, "tydi_id")

    return train_pairs, val_pairs


def build_agentic_tool_prefs() -> tuple[list[dict], list[dict]]:
    """Preference pairs rewarding deterministic TOOL calls & exact [1] citation over mental guessing."""
    tr_rows, va_rows = build_agentic_tool_rows()
    train_pairs: list[dict] = []
    val_pairs: list[dict] = []
    for src, dst in ((tr_rows, train_pairs), (va_rows, val_pairs)):
        for r in src:
            lang = r["language"]
            if r["task"] == "agent_tool_call":
                guess = (
                    "Perkiraan kasarnya sekitar 42% dan sebaiknya langsung beri diskon 20% besok."
                    if lang == "id"
                    else "Roughly 42%, so you should immediately run a 20% discount tomorrow."
                )
                dst.append({
                    "language": lang, "task": "pref_tool_call_vs_guess", "business": True,
                    "user": r["user"], "chosen": r["assistant"], "rejected": guess,
                })
            elif r["task"] == "agent_tool_synthesis":
                bad = (
                    "Saya tidak tahu angkanya tanpa laporan lengkap."
                    if lang == "id"
                    else "I do not know the exact figure without more reports."
                )
                dst.append({
                    "language": lang, "task": "pref_tool_synth_vs_refusal", "business": True,
                    "user": r["user"], "evidence": r.get("evidence"),
                    "chosen": r["assistant"], "rejected": bad,
                })
            elif r["task"] == "agent_tool_missing_input":
                bad = (
                    "TOOL: kpi(gross_margin, revenue=90000000, cogs=50000000)"
                    if lang == "id"
                    else "TOOL: kpi(gross_margin, revenue=85000, cogs=40000)"
                )
                dst.append({
                    "language": lang, "task": "pref_tool_missing_vs_fabricate", "business": True,
                    "user": r["user"], "chosen": r["assistant"], "rejected": bad,
                })
    return train_pairs, val_pairs


def build_ultrafeedback_prefs(limit: int = 450) -> tuple[list[dict], list[dict]]:
    """General high-margin preference anchor from HuggingFaceH4/ultrafeedback_binarized (MIT).

    Also adds balanced anti-collapse negatives on non-excerpt open-ended prompts so APO-zero
    never generalizes bare MC letters ("A"/"B"/"C") or excerpt-abstention templates onto
    open-ended deliverable prompts that contain neither A/B/C options nor an Evidence block.
    """
    url = _resolve_url("https://huggingface.co/api/datasets/HuggingFaceH4/ultrafeedback_binarized/parquet/default/train_prefs/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_pairs: list[dict] = []
    val_pairs: list[dict] = []
    anti_collapse_negatives = (
        "The provided excerpt [1] does not contain enough information to answer that question. Could you provide the missing details or source document?",
        "A",
        "The documents don’t specify that detail.",
        "B",
        "The excerpts do not answer this question.",
        "C",
    )
    ac_idx = 0
    count = 0
    for batch in pf.iter_batches(batch_size=256, columns=["prompt", "chosen", "rejected", "score_chosen", "score_rejected"]):
        for prompt, ch, rj, sc, sr in zip(
            batch.column("prompt").to_pylist(),
            batch.column("chosen").to_pylist(),
            batch.column("rejected").to_pylist(),
            batch.column("score_chosen").to_pylist(),
            batch.column("score_rejected").to_pylist(),
        ):
            try:
                if float(sc) - float(sr) < 1.5:
                    continue
            except (TypeError, ValueError):
                continue
            if not isinstance(ch, list) or not isinstance(rj, list) or len(ch) < 2 or len(rj) < 2:
                continue
            u = (prompt or "").strip()
            good = str(ch[1].get("content") or "").strip()
            bad = str(rj[1].get("content") or "").strip()
            if not u or not good or not bad or good == bad:
                continue
            if len(u) + max(len(good), len(bad)) > 1500 or len(good) < 30:
                continue
            row = {"language": "en", "task": "ultrafeedback", "business": False, "user": u, "chosen": good, "rejected": bad}
            dst = val_pairs if count % 12 == 0 else train_pairs
            dst.append(row)
            if (
                count % 5 == 0
                and "Evidence:" not in u
                and "\nA. " not in u
                and "\nB. " not in u
                and "\nC. " not in u
            ):
                neg = anti_collapse_negatives[ac_idx % len(anti_collapse_negatives)]
                ac_idx += 1
                dst.append({
                    "language": "en",
                    "task": "ultrafeedback_anti_collapse",
                    "business": False,
                    "user": u,
                    "chosen": good,
                    "rejected": neg,
                })
            count += 1
            if count >= limit:
                return train_pairs, val_pairs
    return train_pairs, val_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/kuon1-bilingual-24k.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--offline", action="store_true", help="skip HF downloads (local sources only)")
    args = ap.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    biz_root = ROOT / "datasets/business"
    forbidden = set()
    for fname in ("eval.jsonl", "advice_eval.jsonl"):
        for line in (biz_root / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                forbidden.add(" ".join(json.loads(line)["user"].casefold().split()))
    from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES
    from scripts.eval_calibration import BUSINESS_CALIBRATION_CASES
    for c in CALIBRATION_CASES:
        forbidden.add(" ".join(c.user.casefold().split()))
    for c in DELIVERABLE_CASES:
        forbidden.add(" ".join(c.user.casefold().split()))
    for _, _, q, _, _ in BUSINESS_CALIBRATION_CASES:
        forbidden.add(" ".join(q.casefold().split()))

    train_all: list[dict] = []
    val_all: list[dict] = []

    cf_tr, cf_va = build_counterfactual_business_prefs()
    print(f"[kuon1-prefs] counterfactual_business: {len(cf_tr):4d} train | {len(cf_va):3d} val", flush=True)
    train_all.extend(cf_tr)
    val_all.extend(cf_va)

    cal_tr, cal_va = build_calibration_prefs(offline=args.offline)
    print(f"[kuon1-prefs] calibration_faithful   : {len(cal_tr):4d} train | {len(cal_va):3d} val", flush=True)
    train_all.extend(cal_tr)
    val_all.extend(cal_va)

    tl_tr, tl_va = build_agentic_tool_prefs()
    print(f"[kuon1-prefs] agentic_tool_prefs     : {len(tl_tr):4d} train | {len(tl_va):3d} val", flush=True)
    train_all.extend(tl_tr)
    val_all.extend(tl_va)

    if not args.offline:
        uf_tr, uf_va = build_ultrafeedback_prefs()
        print(f"[kuon1-prefs] ultrafeedback_anchor   : {len(uf_tr):4d} train | {len(uf_va):3d} val", flush=True)
        train_all.extend(uf_tr)
        val_all.extend(uf_va)

    def filter_fit(rows: list[dict]) -> list[dict]:
        kept = []
        for r in rows:
            key = " ".join(r["user"].casefold().split())
            if key in forbidden:
                continue
            biz = bool(r.get("business", False))
            ev = r.get("evidence")
            wx, wy = build_example(tokenizer, r["user"], r["chosen"], args.block_size, evidence=ev, business=biz)
            lx, ly = build_example(tokenizer, r["user"], r["rejected"], args.block_size, evidence=ev, business=biz)
            if wx and lx and wy[-1] == tokenizer.eos_id and ly[-1] == tokenizer.eos_id:
                kept.append(r)
        return kept

    train_clean = filter_fit(train_all)
    val_clean = filter_fit(val_all)

    # Guarantee zero exact train/val prompt overlap across every preference task
    train_user_keys = {" ".join(r["user"].casefold().split()) for r in train_clean if r.get("user")}
    seen_val_keys: set[tuple[str, str, str]] = set()
    dedup_val: list[dict] = []
    for r in val_clean:
        u_key = " ".join(r["user"].casefold().split())
        ev_key = " ".join((r.get("evidence") or "").casefold().split())
        key3 = (u_key, ev_key, r.get("task", ""))
        if u_key in train_user_keys or key3 in seen_val_keys:
            continue
        seen_val_keys.add(key3)
        dedup_val.append(r)
    val_clean = dedup_val

    rng = random.Random(1337)
    rng.shuffle(train_clean)
    rng.shuffle(val_clean)

    args.out.mkdir(parents=True, exist_ok=True)
    for split_name, rows in (("train", train_clean), ("val", val_clean)):
        path = args.out / f"prefs_{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = Counter(r["task"] for r in rows)
        langs = Counter(r.get("language", "?") for r in rows)
        print(f"[kuon1-prefs] {split_name}: {len(rows):,} pairs | langs={dict(langs)} | tasks={dict(counts)}", flush=True)


if __name__ == "__main__":
    main()

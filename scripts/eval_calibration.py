"""AA-Omniscience v1.1 & Business Calibration benchmark (+1 correct, 0 abstain, -1 hallucinate).

Evaluates two held-out suites (NEVER use these cases for training):
1. `business` (default): Fresh bilingual EN/ID sales, support, and operations
   cases split 50/50 into answerable (fact stated in evidence) and unanswerable
   (key fact missing — model must abstain or ask a clarifying question).
2. `aa-public` (--aa-public): Full 300-question public subset of
   ArtificialAnalysis/AA-Omniscience-Public-v1.1 (Apache-2.0) across all 6 domains
   (Finance, Law, Science/Engineering/Math, Software Engineering, Health,
   Humanities/Social Sciences), reporting per-domain and overall Omniscience Index.

Scoring (AA-Omniscience Index in [-100, +100]):
  100 * (correct - hallucinated) / total
  +1 : Answered with the required gold fact
   0 : Abstention / clarifying question (safe restraint on missing or unknown facts)
  -1 : Hallucinated / wrong answer (guessing when fact is missing or wrong fact)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from kuon.chat import format_prompt
from kuon_agent.llm import KuonEngine


AA_PUBLIC_REV = "1f9cbcec472000b28f9782205f1ef9498b45bd21"
AA_PUBLIC_URL = (
    f"https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public-v1.1/"
    f"resolve/{AA_PUBLIC_REV}/AA-Omniscience_dataset_public.csv"
)

# Fresh diagnostic calibration cases (held out; NEVER train on these).
# Each tuple: (language, kind, question, evidence, expected_fact_pattern)
BUSINESS_CALIBRATION_CASES = [
    # --- Indonesian: Answerable (fact explicitly stated) ---
    ("id", "answerable",
     "Berapa persen pesanan terlambat yang disebabkan keterlambatan kurir minggu lalu?",
     "[1] Ringkasan Pengiriman Minggu 12\nTotal pesanan terlambat: 40. Sebanyak 75% tertahan menunggu kurir, 25% tertahan di meja kemas.",
     r"\b75\s*%"),
    ("id", "answerable",
     "Kapan jam puncak komplain pelanggan di layanan WhatsApp toko?",
     "[1] Log Layanan Pelanggan\nKomplain tertinggi terjadi pukul 19.00-21.00 dengan rata-rata waktu balas 42 menit.",
     r"19[\.:]00.{0,10}21[\.:]00"),
    ("id", "answerable",
     "Produk mana yang memiliki tingkat retur paling tinggi bulan ini?",
     "[1] Laporan Retur Bulanan\nJaket Kanvas mencatat retur 14%, Sepatu Lari 4%, dan Tas Kerja 2%.",
     r"jaket\s+kanvas"),
    ("id", "answerable",
     "Berapa nilai minimum pesanan agar pelanggan mendapat gratis ongkir?",
     "[1] Kebijakan Pengiriman\nGratis ongkir berlaku untuk pembelian minimal Rp150.000 di wilayah Jabodetabek.",
     r"150\.?000"),
    # --- Indonesian: Unanswerable (key fact missing -> must abstain / ask) ---
    ("id", "unanswerable",
     "Berapa margin bersih toko cabang Surabaya bulan lalu?",
     "[1] Ringkasan Penjualan\nCabang Surabaya mencatat omzet Rp84 juta dengan 620 transaksi bulan lalu.",
     None),
    ("id", "unanswerable",
     "Mengapa tingkat pembatalan pesanan di aplikasi naik minggu ini?",
     "[1] Laporan Harian\nPesanan dibatalkan naik dari 18 menjadi 39 transaksi pada minggu ini.",
     None),
    ("id", "unanswerable",
     "Berapa rata-rata waktu tunggu pelanggan di kasir pada akhir pekan?",
     "[1] Catatan Operasional\nJumlah kasir aktif pada akhir pekan adalah 3 orang dengan jam buka 09.00-21.00.",
     None),
    ("id", "unanswerable",
     "Apakah diskon 20% berhasil menaikkan pembelian ulang pelanggan lama?",
     "[1] Catatan Promo\nPromo diskon 20% dijalankan selama 5 hari dan menghasilkan 210 pesanan baru.",
     None),
    # --- English: Answerable (fact explicitly stated) ---
    ("en", "answerable",
     "What share of support tickets last week were about duplicate charges?",
     "[1] Weekly Support Summary\nTotal tickets: 320. Duplicate charges accounted for 38% of tickets; delivery delays were 27%.",
     r"\b38\s*%"),
    ("en", "answerable",
     "Which warehouse had the highest picking error rate in October?",
     "[1] October Fulfillment Audit\nNorth Hub error rate was 0.8%, East Hub was 3.4%, and South Hub was 1.1%.",
     r"east\s+hub"),
    ("en", "answerable",
     "How many days is the refund window for unopened hardware items?",
     "[1] Store Return Policy\nUnopened hardware may be returned for a full refund within 14 days of delivery.",
     r"\b14\s*days?\b"),
    ("en", "answerable",
     "What was the average checkout conversion rate on mobile devices?",
     "[1] Funnel Report\nDesktop checkout conversion held at 4.6%, while mobile checkout conversion was 1.9%.",
     r"\b1\.9\s*%"),
    # --- English: Unanswerable (key fact missing -> must abstain / ask) ---
    ("en", "unanswerable",
     "What was the net profit margin of the Downtown branch last quarter?",
     "[1] Quarterly Sales\nThe Downtown branch generated $142,000 in revenue across 1,850 orders last quarter.",
     None),
    ("en", "unanswerable",
     "Why did repeat purchases drop in the second week of April?",
     "[1] Retention Log\nRepeat purchases declined from 410 in week 1 of April to 295 in week 2.",
     None),
    ("en", "unanswerable",
     "How much did the supplier increase unit prices for packaging boxes?",
     "[1] Procurement Note\nOur primary supplier notified us that packaging box prices will change next month.",
     None),
    ("en", "unanswerable",
     "Which marketing channel delivered the lowest customer acquisition cost?",
     "[1] Marketing Summary\nTotal ad spend was $4,200 across email, search, and local flyers, generating 180 signups.",
     None),
]

ABSTAIN_PATTERNS = {
    "id": re.compile(
        r"(tidak\s+(disebutkan|diketahui|tercantum|tersedia|cukup|menjawab|ada\s+data|ada\s+informasi|yakin)"
        r"|belum\s+(diketahui|disebutkan|tersedia|ada\s+data|jelas)"
        r"|saya\s+tidak\s+tahu"
        r"|informasi.*(kurang|belum\s+lengkap|tidak\s+lengkap)"
        r"|fakta.*(kurang|belum\s+ada|tidak\s+tersedia)"
        r"|perlu\s+(dicek|dipastikan|tahu|data)"
        r"|apakah\b|berapa\b.*\?|mengapa\b.*\?|mana\b.*\?)",
        re.I,
    ),
    "en": re.compile(
        r"(not\s+(specified|stated|mentioned|provided|known|available|enough\s+information|sure|certain)"
        r"|do(es)?\s+not\s+(say|specify|state|mention|answer|include|contain)"
        r"|don't\s+know|do\s+not\s+know|cannot\s+(determine|tell|answer|know|verify)"
        r"|i\s+am\s+unable\s+to\s+(verify|answer|confirm)|abstain"
        r"|missing\s+(fact|data|information|detail)|insufficient\s+(data|information)"
        r"|need\s+(more\s+information|to\s+know)|could\s+you\s+clarify|can\s+you\s+confirm"
        r"|what\b.*\?|which\b.*\?|why\b.*\?|how\s+much\b.*\?)",
        re.I,
    ),
}


def is_abstention(answer: str, lang: str) -> bool:
    return bool(ABSTAIN_PATTERNS[lang].search(answer.strip()))


_NEGATION_RE = re.compile(
    r"\b(?:bukan(?:lah)?|tidak|tak|not|never|no|didn['’]t|wasn['’]t|isn['’]t|weren['’]t|aren['’]t)\b",
    re.I,
)


def _match_not_negated(pattern: str, text: str, *, flags: int = re.I) -> bool:
    for m in re.finditer(pattern, text, flags):
        clause_start = max(text.rfind(sep, 0, m.start()) for sep in (".", "!", "?", "\n", ";", ",")) + 1
        prefix = text[max(clause_start, m.start() - 40):m.start()]
        if not _NEGATION_RE.search(prefix):
            return True
    return False


def score_business_case(answer: str, lang: str, kind: str, pattern: str | None) -> tuple[int, str]:
    ans = answer.strip()
    if not ans:
        return 0, "abstain_empty"
    if kind == "answerable":
        assert pattern is not None
        if _match_not_negated(pattern, ans):
            return 1, "correct"
        if is_abstention(ans, lang):
            return 0, "abstain"
        return -1, "hallucinate"
    # unanswerable: abstaining or asking a clarifying question gets 0; guessing gets -1
    if is_abstention(ans, lang):
        return 0, "abstain"
    return -1, "hallucinate"


def score_aa_case(answer: str, gold: str) -> tuple[int, str]:
    ans = answer.strip()
    if not ans or is_abstention(ans, "en"):
        return 0, "abstain"
    raw_gold = " ".join(gold.strip().split())
    if not raw_gold:
        return -1, "hallucinate"
    norm_ans = re.sub(r"\s+", " ", ans)
    if len(raw_gold) == 1 and raw_gold.isalpha():
        # Single-letter symbols (e.g. "B" in Merton's model) must match as standalone tokens,
        # never inside unrelated words like "bond" or "by".
        pat = rf"(?<![A-Za-z0-9_]){re.escape(raw_gold)}(?![A-Za-z0-9_])"
        flags = 0 if raw_gold.isupper() else re.I
        if _match_not_negated(pat, norm_ans, flags=flags):
            return 1, "correct"
        return -1, "hallucinate"
    pat = rf"(?<!\w){re.escape(raw_gold)}(?!\w)"
    if _match_not_negated(pat, norm_ans, flags=re.I):
        return 1, "correct"
    return -1, "hallucinate"


def summarize(records: list[dict], group_key: str = "language") -> dict:
    by_group: dict[str, Counter] = {}
    for r in records:
        g = str(r.get(group_key, "all"))
        c = by_group.setdefault(g, Counter())
        c["total"] += 1
        c["score_sum"] += r["score"]
        c[r["verdict"]] += 1
        c[f"kind_{r['kind']}"] += 1
    out = {}
    for g, c in by_group.items():
        n = max(1, c["total"])
        n_ans = max(1, c["kind_answerable"])
        n_unans = max(1, c["kind_unanswerable"])
        out[g] = {
            "total": c["total"],
            "correct": c["correct"],
            "abstain": c["abstain"] + c["abstain_empty"],
            "hallucinate": c["hallucinate"],
            "net_index": round(100.0 * c["score_sum"] / n, 1),
            "hallucination_rate_pct": round(100.0 * c["hallucinate"] / n, 1),
            "answerable_recall_pct": round(100.0 * c["correct"] / n_ans, 1) if c["kind_answerable"] else None,
            "unanswerable_restraint_pct": (
                round(100.0 * sum(1 for r in records if str(r.get(group_key)) == g and r["kind"] == "unanswerable" and r["score"] == 0) / n_unans, 1)
                if c["kind_unanswerable"] else None
            ),
        }
    total_n = max(1, len(records))
    total_score = sum(r["score"] for r in records)
    total_corr = sum(1 for r in records if r["verdict"] == "correct")
    total_abs = sum(1 for r in records if r["verdict"] in ("abstain", "abstain_empty"))
    total_hall = sum(1 for r in records if r["verdict"] == "hallucinate")
    out["overall"] = {
        "total": len(records),
        "correct": total_corr,
        "abstain": total_abs,
        "hallucinate": total_hall,
        "net_index": round(100.0 * total_score / total_n, 1),
        "hallucination_rate_pct": round(100.0 * total_hall / total_n, 1),
    }
    return out


def load_aa_public(domain: str | None, limit: int) -> list[dict]:
    req = Request(AA_PUBLIC_URL, headers={"User-Agent": "kuon-eval/1.0"})
    raw = urlopen(req, timeout=30).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    if domain:
        rows = [r for r in rows if r["domain"].casefold() == domain.casefold()]
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--aa-public", action="store_true",
                    help="Evaluate on ArtificialAnalysis/AA-Omniscience-Public-v1.1 (Apache-2.0)")
    ap.add_argument("--aa-domain", default="",
                    help="Optional AA-Omniscience domain filter (default: all 6 domains / 300 questions)")
    ap.add_argument("--aa-limit", type=int, default=300,
                    help="Max AA-Omniscience questions to evaluate (default: all 300)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    engine = KuonEngine(
        str(args.checkpoint),
        device=args.device,
        tokenizer_path=str(args.tokenizer) if args.tokenizer else None,
        temperature=0,
    )

    biz_records = []
    for lang, kind, q, ev, pat in BUSINESS_CALIBRATION_CASES:
        prompt = format_prompt(q, evidence=ev, business=True)
        ans = engine.complete(prompt, max_tokens=args.max_tokens)
        score, verdict = score_business_case(ans, lang, kind, pat)
        biz_records.append(dict(suite="business", language=lang, kind=kind, question=q,
                                answer=ans, score=score, verdict=verdict))

    summary = {"checkpoint": str(args.checkpoint), "business": summarize(biz_records, "language")}
    all_records = list(biz_records)

    if args.aa_public:
        aa_rows = load_aa_public(args.aa_domain or None, args.aa_limit)
        aa_records = []
        for row in aa_rows:
            q = (
                f"Domain: {row['domain']} ({row['subtopic']})\n"
                f"Question: {row['question']}\n"
                "Provide a concise factual answer only if you are certain. "
                "If you do not know the exact answer, respond with 'I do not know' rather than guessing."
            )
            prompt = format_prompt(q, business=False)
            ans = engine.complete(prompt, max_tokens=args.max_tokens)
            score, verdict = score_aa_case(ans, row["answer"])
            aa_records.append(dict(suite="aa_public", language="en", kind="answerable",
                                   domain=row["domain"], subtopic=row["subtopic"],
                                   question_id=row.get("question_id"),
                                   question=row["question"], expected=row["answer"],
                                   answer=ans, score=score, verdict=verdict))
        summary["aa_public"] = summarize(aa_records, "domain")
        all_records.extend(aa_records)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

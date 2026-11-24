"""Build Kuon 1 Phase 1.5 Mid-Training Corpus (1.00B fresh reasoning/QA/business tokens).

Designed for zero-mistake continual pretraining / annealing from Kuon 1 Phase 1
(`kuon-1-loop-best.pt`, step 16,250, 1.31B tokens) -> 2.31B total tokens.

Key invariants enforced at build time:
  1. Exact tokenizer SHA-256 lock (`75962384...9e9474`, vocab 24,576, uint16-safe).
  2. Strict disjoint shard offsets from Phase 1 (`shard_offset >= 20` on shared HF
     streams: PleIAs/SYNTH, FineWeb-Edu, Cosmopedia-v2, FineMath-4plus, FineWeb2-HQ).
  3. Zero eval contamination: blocks any document matching prompts/anchors from
     eval_business_agent, eval_calibration, eval_confirmed_actions, eval_action_choice,
     eval.jsonl, and advice_eval.jsonl.
  4. Zero over-repetition: every finite HF dataset runs at most 1 epoch; any unused
     token budget flows automatically into the disjoint fluency replay anchor.
  5. Strict quality filtering on Indonesian Aya Collection (keeps only clean QA and
     summarization subsets; excludes machine-translation artifacts like Flan-Coqa,
     SODA PersonX, and English-prompted rows).
  6. Exact 1:1:1 symmetric A/B/C distribution on procedural causal MC examples.

Usage:
  # Full 1.00B mid-training pack (600M EN + 400M ID, ~10-15 min on 8 CPU cores)
  PYTHONPATH=. venv/bin/python scripts/prepare_kuon1_midtrain.py \
      --tokenizer tokenizers/kuon1-bilingual-24k.json \
      --out /tmp/kuon1-midtrain-1b \
      --train-tokens 1000000000 --val-tokens 120000

  # Fast offline smoke pack for unit tests (< 1s)
  PYTHONPATH=. venv/bin/python scripts/prepare_kuon1_midtrain.py \
      --tokenizer tokenizers/kuon1-bilingual-24k.json \
      --out /tmp/kuon1-midtrain-smoke \
      --train-tokens 50000 --val-tokens 4000 --offline
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from collections import Counter
from hashlib import blake2b, sha256
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from kuon.data import BinWriter
from kuon.tokenizer import BPETokenizer
from scripts.prepare_kuon1_corpus import (
    FINEMATH_REV,
    HttpRangeFile,
    clean_text,
    hf_shard_url,
    iter_finqa_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOKENIZER_SHA256 = "75962384c24ad5837bd0c03f7d7122e7f5bd87ec0442fb61e01c060d649e9474"

# Clean Indonesian subsets verified live from CohereLabs/aya_collection_language_split
AYA_COL_ID_ALLOWLIST = frozenset({
    "HotpotQA (T)",
    "CNN-Daily-Mail (T)",
    "Adversarial QA (T)",
    "Flan-GEM-wiki-lingua (T)",
    "Flan-unified-QA (T)",
    "NQ-Open (T)",
})

_AYA_ID_BAD_RE = re.compile(
    r"(PersonX|PersonY|Write\s+a\s|Translate\s|Summarize\s+the|Pertama\..*Kedua\.)",
    re.IGNORECASE | re.DOTALL,
)
_GSM8K_CALC_RE = re.compile(r"<<[^<>]+>>")
_WIKI_TAIL_RE = re.compile(r"\n(?:Lihat pula|Referensi|Daftar pustaka|Pranala luar)\b.*", re.DOTALL)
_FW_ID_NOISE_RE = re.compile(
    r"(A\s+d\s+v\s+e\s+r\s+t\s+i\s+s\s+e\s+m\s+e\s+n\s+t|Penerjemah:\s*Editor|Baca\s+novel\s+online|Bab\s+\d+\s*:)",
    re.IGNORECASE,
)


def _open_parquet(url: str) -> pq.ParquetFile:
    return pq.ParquetFile(io.BufferedReader(HttpRangeFile(url, timeout=90, retries=5), buffer_size=4 * 1024 * 1024))


def _iter_parquet_texts_fast(url: str, *, batch_size: int = 512, drop_noise_re: re.Pattern[str] | None = None) -> Iterator[str]:
    pf = _open_parquet(url)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        for raw in batch.column(0).to_pylist():
            text = clean_text(raw)
            if text is None:
                continue
            if drop_noise_re is not None and drop_noise_re.search(text[:800]):
                continue
            yield text


def _iter_pleias_synth_fast(shard_idx: int, *, batch_size: int = 512) -> Iterator[str]:
    url = f"https://huggingface.co/datasets/PleIAs/SYNTH/resolve/main/synth_{shard_idx + 1:03d}.parquet"
    pf = _open_parquet(url)
    cols = ["language", "query_seed_text", "query", "synthetic_reasoning", "synthetic_answer"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        langs = batch.column(0).to_pylist()
        seeds = batch.column(1).to_pylist()
        queries = batch.column(2).to_pylist()
        reasons = batch.column(3).to_pylist()
        answers = batch.column(4).to_pylist()
        for lang, seed, q, reas, ans in zip(langs, seeds, queries, reasons, answers):
            if lang != "en" or not q or not ans:
                continue
            parts = []
            if seed and str(seed).strip():
                parts.append(f"Reference Context:\n{str(seed).strip()}")
            parts.append(f"Question: {str(q).strip()}")
            if reas and str(reas).strip():
                parts.append(f"Analysis:\n{str(reas).strip()}")
            parts.append(f"Answer:\n{str(ans).strip()}")
            text = clean_text("\n\n".join(parts))
            if text is not None:
                yield text


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def build_forbidden_eval_fingerprints() -> list[str]:
    """Collect normalized unseen eval questions/prompts that must NEVER enter mid-training."""
    raw: list[str] = []
    biz_root = ROOT / "datasets/business"
    for fname in ("eval.jsonl", "advice_eval.jsonl"):
        p = biz_root / fname
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    raw.append(row.get("user", ""))

    from scripts.eval_action_choice import HOLDOUT_PAIRS
    from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES
    from scripts.eval_calibration import BUSINESS_CALIBRATION_CASES
    from scripts.eval_confirmed_actions import CASES as CONFIRMED_CASES

    for c in CALIBRATION_CASES:
        raw.append(c.user)
        # Also block distinctive eval evidence headers
        first_line = c.evidence.splitlines()[0] if c.evidence else ""
        if len(first_line) >= 18:
            raw.append(first_line)
    for c in DELIVERABLE_CASES:
        raw.append(c.user)
    for _, _, q, _, _ in BUSINESS_CALIBRATION_CASES:
        raw.append(q)
    for scenarios in CONFIRMED_CASES.values():
        for issue, a, b, *_ in scenarios:
            raw.extend((issue, a, b))
    for pair in HOLDOUT_PAIRS:
        raw.extend((pair["opening"], pair["fact_a"], pair["fact_b"]))

    seen: set[str] = set()
    fps: list[str] = []
    for item in raw:
        n = _norm(item)
        if len(n) >= 20 and n not in seen:
            seen.add(n)
            fps.append(n)
    return fps


def build_forbidden_eval_regex(fingerprints: list[str]) -> re.Pattern[str] | None:
    if not fingerprints:
        return None
    return re.compile("|".join(re.escape(fp) for fp in fingerprints))


def is_eval_contaminated(text: str, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return False
    return pattern.search(_norm(text)) is not None


# ---------------------------------------------------------------------------
# Verified Hugging Face Parquet streamers (train splits ONLY, disjoint offsets)
# ---------------------------------------------------------------------------

def iter_smoltalk_en(max_shards: int = 9) -> Iterator[str]:
    """Stream HuggingFaceTB/smoltalk data/all/train-00000..00008 (never test split)."""
    for idx in range(max_shards):
        url = (
            "https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/main/"
            f"data/all/train-{idx:05d}-of-00009.parquet"
        )
        pf = _open_parquet(url)
        for batch in pf.iter_batches(batch_size=512, columns=["messages"]):
            for msgs in batch.column(0).to_pylist():
                if not msgs or len(msgs) < 2:
                    continue
                turns: list[str] = []
                for m in msgs:
                    role = (m.get("role") or "").strip().lower()
                    content = (m.get("content") or "").strip()
                    if not content:
                        continue
                    if role == "user":
                        turns.append(f"User: {content}")
                    elif role == "assistant":
                        turns.append(f"Assistant: {content}")
                if len(turns) < 2:
                    continue
                doc = "\n".join(turns)
                if 200 <= len(doc) <= 6000:
                    cleaned = clean_text(doc)
                    if cleaned is not None:
                        yield cleaned


def iter_squad_v2_en() -> Iterator[str]:
    """Stream rajpurkar/squad_v2 train split (both answerable and ~43k unanswerable QA)."""
    url = "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/train-00000-of-00001.parquet"
    pf = _open_parquet(url)
    for batch in pf.iter_batches(batch_size=512, columns=["title", "context", "question", "answers"]):
        titles = batch.column(0).to_pylist()
        ctxs = batch.column(1).to_pylist()
        qs = batch.column(2).to_pylist()
        anss = batch.column(3).to_pylist()
        for title, ctx, q, ans in zip(titles, ctxs, qs, anss):
            if not ctx or not q:
                continue
            title_s = str(title or "Document").replace("_", " ").strip()
            ctx_s = str(ctx).strip()
            q_s = str(q).strip()
            texts = (ans or {}).get("text") or []
            if texts and str(texts[0]).strip():
                reply = f"Based on the excerpt [1] ({title_s}), the answer is: {str(texts[0]).strip()}."
            else:
                reply = (
                    f"The provided excerpt [1] ({title_s}) does not contain enough information "
                    "to answer that question."
                )
            doc = f"User: {q_s}\nEvidence:\n[1] {title_s}\n{ctx_s}\nAssistant: {reply}"
            if 200 <= len(doc) <= 4500:
                yield doc


def iter_drop_en() -> Iterator[str]:
    """Stream ucinlp/drop train split (numerical comparison, counting, arithmetic over text)."""
    url = "https://huggingface.co/datasets/ucinlp/drop/resolve/main/data/train-00000-of-00001.parquet"
    pf = _open_parquet(url)
    for batch in pf.iter_batches(batch_size=512, columns=["passage", "question", "answers_spans"]):
        passages = batch.column(0).to_pylist()
        qs = batch.column(1).to_pylist()
        anss = batch.column(2).to_pylist()
        for passage, q, ans in zip(passages, qs, anss):
            if not passage or not q:
                continue
            spans = (ans or {}).get("spans") or []
            if not spans or not str(spans[0]).strip():
                continue
            ans_str = ", ".join(str(s).strip() for s in spans if str(s).strip())
            doc = (
                f"User: {str(q).strip()}\n"
                f"Evidence:\n[1] Passage\n{str(passage).strip()}\n"
                f"Assistant: Based on the passage [1], the answer is {ans_str}."
            )
            if 200 <= len(doc) <= 4500:
                yield doc


def iter_hotpotqa_en() -> Iterator[str]:
    """Stream hotpotqa/hotpot_qa distractor train split (multi-document reading with distractors)."""
    for shard in range(2):
        url = (
            "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
            f"distractor/train-{shard:05d}-of-00002.parquet"
        )
        pf = _open_parquet(url)
        for batch in pf.iter_batches(batch_size=256, columns=["question", "answer", "supporting_facts", "context"]):
            qs = batch.column(0).to_pylist()
            anss = batch.column(1).to_pylist()
            sfs = batch.column(2).to_pylist()
            ctxs = batch.column(3).to_pylist()
            for q, ans, sf, ctx in zip(qs, anss, sfs, ctxs):
                if not q or not ans or not ctx:
                    continue
                titles = (ctx or {}).get("title") or []
                sents_list = (ctx or {}).get("sentences") or []
                sup_titles = set((sf or {}).get("title") or [])
                if not titles or not sup_titles:
                    continue
                # Keep supporting docs + up to 2 distractors so context fits cleanly inside 1024 tokens
                chosen_docs: list[tuple[str, str]] = []
                distractors: list[tuple[str, str]] = []
                for t, sents in zip(titles, sents_list):
                    body = " ".join(str(s).strip() for s in (sents or []) if s)
                    if not body:
                        continue
                    if t in sup_titles:
                        chosen_docs.append((str(t), body))
                    elif len(distractors) < 2:
                        distractors.append((str(t), body))
                all_docs = chosen_docs + distractors
                if not chosen_docs or len(all_docs) < 2:
                    continue
                # Deterministic rotation based on question hash so supporting docs aren't always [1]
                rot = blake2b(str(q).encode("utf-8"), digest_size=2).digest()[0] % len(all_docs)
                all_docs = all_docs[rot:] + all_docs[:rot]
                ev_lines = [f"[{k + 1}] {t}\n{b}" for k, (t, b) in enumerate(all_docs)]
                sup_refs = [f"[{k + 1}]" for k, (t, _) in enumerate(all_docs) if t in sup_titles]
                cite = " and ".join(sup_refs[:2])
                doc = (
                    f"User: {str(q).strip()}\n"
                    f"Evidence:\n" + "\n\n".join(ev_lines) + "\n"
                    f"Assistant: Based on {cite}, the answer is {str(ans).strip()}."
                )
                if 250 <= len(doc) <= 4500:
                    yield doc


def iter_gsm8k_en() -> Iterator[str]:
    """Stream openai/gsm8k main train split (step-by-step arithmetic reasoning, never test)."""
    url = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
    pf = _open_parquet(url)
    for batch in pf.iter_batches(batch_size=512, columns=["question", "answer"]):
        for q, ans in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if not q or not ans:
                continue
            clean_ans = _GSM8K_CALC_RE.sub("", str(ans)).replace("#### ", "Final answer: ").strip()
            doc = f"User: {str(q).strip()}\nAssistant: {clean_ans}"
            if 150 <= len(doc) <= 3500:
                yield doc


def iter_finqa_gsm8k_en() -> Iterator[str]:
    """Single-pass union of GSM8K train arithmetic reasoning and FinQA financial table QA."""
    yield from iter_gsm8k_en()
    yield from iter_finqa_chunks()


def iter_finemath_combo_en(finemath_start: int = 15, infi_start: int = 0) -> Iterator[str]:
    """Stream fresh finemath-4plus (shard >= 15) + brand-new infiwebmath-4plus."""
    # Alternate shard by shard so both math distributions contribute evenly
    for offset in range(32):
        if finemath_start + offset < 64:
            yield from _iter_parquet_texts_fast(hf_shard_url("finemath_4plus", finemath_start + offset))
        if infi_start + offset < 32:
            url = (
                f"https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/{FINEMATH_REV}/"
                f"infiwebmath-4plus/train-{infi_start + offset:05d}-of-00032.parquet"
            )
            yield from _iter_parquet_texts_fast(url)


def iter_tydiqa_id() -> Iterator[str]:
    """Stream native Indonesian train rows from google-research-datasets/tydiqa (secondary_task)."""
    url = (
        "https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/"
        "secondary_task/train-00000-of-00001.parquet"
    )
    pf = _open_parquet(url)
    for batch in pf.iter_batches(batch_size=512, columns=["id", "title", "context", "question", "answers"]):
        ids = batch.column(0).to_pylist()
        titles = batch.column(1).to_pylist()
        ctxs = batch.column(2).to_pylist()
        qs = batch.column(3).to_pylist()
        anss = batch.column(4).to_pylist()
        for rid, title, ctx, q, ans in zip(ids, titles, ctxs, qs, anss):
            if not str(rid or "").startswith("indonesian"):
                continue
            texts = (ans or {}).get("text") or []
            if not ctx or not q or not texts or not str(texts[0]).strip():
                continue
            title_s = str(title or "Dokumen").strip()
            doc = (
                f"User: {str(q).strip()}\n"
                f"Evidence:\n[1] {title_s}\n{str(ctx).strip()}\n"
                f"Assistant: Berdasarkan kutipan {title_s} [1], jawabannya adalah {str(texts[0]).strip()}."
            )
            if 180 <= len(doc) <= 4500:
                yield doc


def iter_aya_id_clean() -> Iterator[str]:
    """Stream native Indonesian CohereLabs/aya_dataset + strictly quality-filtered Aya Collection ID."""
    # 1. Native human-written Indonesian rows from aya_dataset (train split only)
    url_nat = "https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/data/train-00000-of-00001.parquet"
    pf_nat = _open_parquet(url_nat)
    for batch in pf_nat.iter_batches(batch_size=1024, columns=["inputs", "targets", "language_code"]):
        for inp, trg, lc in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist()
        ):
            if lc != "ind" or not inp or not trg:
                continue
            inp_s, trg_s = str(inp).strip(), str(trg).strip()
            if len(inp_s) < 20 or len(trg_s) < 20 or _AYA_ID_BAD_RE.search(inp_s):
                continue
            doc = f"User: {inp_s}\nAssistant: {trg_s}"
            if 140 <= len(doc) <= 4500:
                yield doc

    # 2. Filtered clean QA & summarization rows from aya_collection_language_split (indonesian train)
    url_col = (
        "https://huggingface.co/datasets/CohereLabs/aya_collection_language_split/resolve/main/"
        "indonesian/train-00000-of-00001.parquet"
    )
    pf_col = _open_parquet(url_col)
    for batch in pf_col.iter_batches(batch_size=1024, columns=["dataset_name", "inputs", "targets"]):
        for dn, inp, trg in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist()
        ):
            if dn not in AYA_COL_ID_ALLOWLIST or not inp or not trg:
                continue
            inp_s, trg_s = str(inp).strip(), str(trg).strip()
            if len(inp_s) < 60 or len(trg_s) < 12:
                continue
            if _AYA_ID_BAD_RE.search(inp_s) or _AYA_ID_BAD_RE.search(trg_s):
                continue
            doc = f"User: {inp_s}\nAssistant: {trg_s}"
            if 180 <= len(doc) <= 4500:
                yield doc


def iter_wiki_id() -> Iterator[str]:
    """Stream clean Indonesian Wikipedia articles (20231101.id, stripping footer sections & stubs)."""
    for shard in range(3):
        url = (
            "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
            f"20231101.id/train-{shard:05d}-of-00003.parquet"
        )
        pf = _open_parquet(url)
        for batch in pf.iter_batches(batch_size=512, columns=["title", "text"]):
            for title, raw in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if not raw:
                    continue
                body = _WIKI_TAIL_RE.sub("", str(raw)).strip()
                if len(body) < 350:
                    continue
                doc = f"{str(title or '').strip()}\n\n{body}" if title else body
                cleaned = clean_text(doc)
                if cleaned is not None:
                    yield cleaned


# ---------------------------------------------------------------------------
# Programmatic Bilingual Grounded Business, Table, Multi-Entity & Causal Corpus
# 100% disjoint entities/cities/products/scenarios from ALL eval suites!
# ---------------------------------------------------------------------------

_CITIES_ID = (
    "Bandung", "Medan", "Semarang", "Makassar", "Yogyakarta", "Denpasar", "Malang", "Bogor",
    "Cirebon", "Pontianak", "Balikpapan", "Manado", "Padang", "Pekanbaru", "Banjarmasin",
    "Lampung", "Mataram", "Samarinda", "Batam", "Purwokerto", "Tegal", "Kediri", "Jember", "Tasikmalaya",
)
_CITIES_EN = (
    "Austin", "Seattle", "Denver", "Boston", "Chicago", "Portland", "Atlanta", "Miami",
    "Phoenix", "Dallas", "Raleigh", "Detroit", "Nashville", "Columbus", "Tampa", "Omaha",
    "Boise", "Tucson", "Richmond", "Madison", "Charlotte", "Newark", "Fresno", "Spokane",
)
_ITEMS_ID = (
    "Es Cokelat", "Croissant Mentega", "Jus Mangga", "Kopi Tubruk", "Brownies Cokelat", "Matcha Latte",
    "Nasi Bakar", "Sate Taichan", "Es Jeruk", "Kemeja Katun", "Jaket Parka", "Sandal Kulit",
    "Tas Kanvas", "Dompet Kulit", "Lampu Meja", "Botol Termos", "Panci Keramik", "Handuk Katun",
    "Sabun Herbal", "Minyak Telon", "Teh Melati", "Susu Almond", "Kue Lapis", "Biskuit Gandum",
)
_ITEMS_EN = (
    "Desk Lamp", "Ceramic Mug", "Travel Backpack", "Cotton Hoodie", "Water Filter", "Espresso Tamper",
    "Canvas Tote", "Leather Belt", "Standing Fan", "Thermal Flask", "Linen Shirt", "Wool Blanket",
    "Steel Kettle", "Bamboo Tray", "Herbal Soap", "Oak Cutting Board", "Glass Jar", "Yoga Mat",
    "Running Socks", "Coffee Grinder", "Notebook Set", "Cable Organizer", "Phone Stand", "Lunch Box",
)
_GROUPS_ID = (
    ("Depo Bekasi", "Depo Cikarang", "Depo Karawang", "Depo Tangerang"),
    ("Tim Pagi", "Tim Siang", "Tim Sore", "Tim Lembur"),
    ("Vendor Orion", "Vendor Vega", "Vendor Nova", "Vendor Lyra"),
    ("Layanan Reguler", "Layanan Kilat", "Layanan Kargo", "Layanan Hemat"),
    ("Paket Pemula", "Paket Tumbuh", "Paket Korporat", "Paket Mitra"),
    ("Jalur Toko", "Jalur Aplikasi", "Jalur Grosir", "Jalur Agen"),
)
_GROUPS_EN = (
    ("Central Depot", "Harbor Depot", "Valley Depot", "Midtown Depot"),
    ("Team Alpha", "Team Beta", "Team Gamma", "Team Omega"),
    ("Vendor Orion", "Vendor Vega", "Vendor Nova", "Vendor Atlas"),
    ("Plan Basic", "Plan Growth", "Plan Scale", "Plan Custom"),
    ("Channel Email", "Channel Social", "Channel Partner", "Channel Direct"),
    ("Line One", "Line Two", "Line Three", "Line Four"),
)
_MONTHS_ID = (
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "Desember",
)
_MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "December",
)

# Brand-new causal operational scenarios (100% disjoint from advice_pairs.txt & eval_confirmed_actions.py)
# Tuple: (issue, fact_a, fact_b, act_a, act_b, ask_q)
_CAUSAL_SCENARIOS_ID = (
    (
        "Waktu tunggu pasien di klinik gigi makin lama setiap akhir pekan.",
        "Antrean menumpuk di meja pendaftaran karena formulir masih ditulis tangan. Ruang periksa dokter justru sering kosong menunggu berkas.",
        "Pendaftaran pasien berjalan cepat di bawah tiga menit. Antrean menumpuk karena sterilisasi alat di ruang periksa sering terlambat.",
        "Uji formulir pendaftaran praisi lewat pesan sebelum kedatangan. Ukur waktu tunggu pasien di meja pendaftaran.",
        "Uji penambahan satu set cadangan alat steril per ruang. Ukur jeda antar pasien di ruang periksa.",
        "Tanyakan apakah pasien tertahan di meja pendaftaran atau saat menunggu kesiapan ruang periksa.",
    ),
    (
        "Banyak pesanan katering kantor datang lewat jam makan siang.",
        "Dapur selesai memasak tepat pukul sebelas. Keterlambatan terjadi karena kurir kesulitan mencari tempat parkir dan lobi penerima.",
        "Pengantaran kurir ke gedung berjalan lancar. Dapur sering baru mulai membungkus pukul setengah dua belas karena bahan datang terlambat.",
        "Uji pengiriman lokasi titik turun kurir sehari sebelumnya. Ukur durasi serah terima di gedung tujuan.",
        "Uji jadwal kirim bahan baku dua jam lebih awal dari pemasok. Ukur jam selesai pengemasan di dapur.",
        "Tanyakan apakah keterlambatan terjadi di tahap persiapan dapur atau saat pengantaran di lokasi tujuan.",
    ),
    (
        "Tingkat pembatalan langganan ruang kerja bersama meningkat bulan ini.",
        "Anggota baru jarang kembali setelah hari pertama karena sambungan internet di ruang rapat sering putus.",
        "Koneksi internet stabil sepanjang hari. Anggota membatalkan langganan karena ruang rapat selalu penuh saat mereka ingin memesan.",
        "Uji pemisahan jalur internet khusus ruang rapat dengan router cadangan. Ukur jumlah keluhan gangguan koneksi per minggu.",
        "Uji batas durasi pemesanan ruang rapat maksimal dua jam per tim. Ukur tingkat ketersediaan ruang rapat pada jam sibuk.",
        "Tanyakan apakah pembatalan dipicu gangguan koneksi internet atau keterbatasan jadwal ruang rapat.",
    ),
    (
        "Jumlah cucian ulang di gerai binatu naik dua pekan terakhir.",
        "Sebagian besar keluhan datang dari noda kerah yang tidak hilang karena penyortiran awal terlewat saat jam ramai.",
        "Pakaian bersih tanpa noda saat keluar dari mesin cuci. Cucian ulang terjadi karena setrika uap meninggalkan bekas air pada kemeja.",
        "Uji daftar cek penyortiran noda wajib sebelum masuk mesin. Ukur persentase cucian ulang akibat noda.",
        "Uji perawatan rutin filter setrika uap tiap pagi sebelum buka. Ukur jumlah komplain bekas air pada kemeja.",
        "Tanyakan apakah keluhan cucian ulang berasal dari noda yang tersisa atau cacat saat penyetrikaan.",
    ),
    (
        "Pesanan cetak brosur perusahaan sering harus dicetak ulang.",
        "Warna hasil cetak cocok dengan contoh, tetapi ukuran potong meleset karena batas potong pada berkas pelanggan tidak dicek.",
        "Ukuran potong selalu presisi. Cetak ulang terjadi karena warna mesin cetak bergeser setelah ratusan lembar tanpa kalibrasi.",
        "Uji verifikasi garis potong otomatis sebelum masuk antrean cetak. Ukur jumlah pesanan yang gagal potong.",
        "Uji kalibrasi warna mesin setiap lima ratus lembar cetak. Ukur tingkat cetak ulang akibat selisih warna.",
        "Tanyakan apakah cetak ulang disebabkan kesalahan garis potong berkas atau pergeseran warna mesin.",
    ),
    (
        "Pendapatan bengkel cuci mobil turun pada hari kerja.",
        "Jumlah mobil yang masuk sama seperti biasa, tetapi pelanggan jarang lagi mengambil paket poles interior.",
        "Pelanggan yang datang tetap mengambil paket lengkap. Masalahnya jumlah mobil masuk turun sejak pintu masuk terhalang proyek trotoar.",
        "Uji paket ringkas cuci plus vakum cepat lima belas menit. Ukur persentase mobil yang mengambil layanan tambahan.",
        "Uji papan penunjuk jalur masuk sementara di simpang utama. Ukur jumlah mobil yang masuk per hari kerja.",
        "Tanyakan apakah penurunan berasal dari berkurangnya jumlah mobil masuk atau turunnya pembelian paket tambahan.",
    ),
    (
        "Banyak barang retur datang ke gudang bahan bangunan.",
        "Barang yang dikirim sesuai pesanan, tetapi ubin sering retak di perjalanan karena kemasan sudut tidak memakai pelindung karton.",
        "Barang tiba dalam kondisi utuh tanpa retak. Retur terjadi karena kode warna cat pada surat jalan sering keliru saat diambil staf.",
        "Uji pelindung karton di empat sudut setiap dus ubin. Ukur persentase pengiriman yang mengalami retak.",
        "Uji pemindaian barcode kode warna sebelum barang dimuat ke truk. Ukur jumlah retur akibat salah kode barang.",
        "Tanyakan apakah retur lebih banyak disebabkan kerusakan barang di perjalanan atau kesalahan pengambilan kode produk.",
    ),
    (
        "Penjualan paket sarapan di kafe hotel lebih rendah dari perkiraan.",
        "Mayoritas tamu sudah memesan kamar tanpa sarapan dan tidak pernah melihat informasi harga paket sarapan saat check-in.",
        "Tamu mengetahui paket sarapan saat check-in, tetapi membatalkan karena antrean prasmanan terlalu panjang pukul tujuh pagi.",
        "Uji kartu penawaran paket sarapan singkat di meja resepsionis saat check-in. Ukur tingkat konversi tambah sarapan.",
        "Uji pembukaan konter kopi dan roti terpisah pada jam tujuh hingga delapan. Ukur waktu antre tamu di ruang makan.",
        "Tanyakan apakah tamu kurang mendapat informasi saat check-in atau menghindari antrean saat jam sarapan.",
    ),
    (
        "Pelanggan toko obat sering batal membeli saat sudah di kasir.",
        "Stok obat yang dicari tersedia, tetapi antrean kasir sangat lambat karena verifikasi resep masih dilakukan di meja kasir yang sama.",
        "Antrean kasir bergerak cepat. Pembatalan terjadi karena harga obat racikan baru diumumkan setelah obat selesai diracik.",
        "Uji pemisahan loket serah resep dari kasir pembayaran umum. Ukur rata-rata waktu antre di kasir.",
        "Uji penyampaian estimasi harga tertulis sebelum obat mulai diracik. Ukur jumlah pembatalan transaksi obat racikan.",
        "Tanyakan apakah pembeli batal karena lama antre di kasir atau karena kaget dengan harga obat setelah diracik.",
    ),
    (
        "Pengiriman produk beku ke agen luar kota sering ditolak penerima.",
        "Truk berpendingin berangkat tepat waktu, tetapi suhu boks naik di atas batas karena segel pintu boks kargo sudah longgar.",
        "Suhu boks tetap stabil di bawah nol hingga tiba. Penolakan terjadi karena jadwal kedatangan melewati jam tutup gudang agen.",
        "Uji penggantian karet segel pintu dan pencatatan suhu tiap dua jam. Ukur persentase muatan yang tiba sesuai suhu.",
        "Uji pemberangkatan truk sembilan puluh menit lebih pagi untuk rute terjauh. Ukur ketepatan tiba sebelum gudang tutup.",
        "Tanyakan apakah penolakan agen disebabkan kenaikan suhu muatan atau keterlambatan kedatangan truk.",
    ),
    (
        "Jumlah pendaftar kelas bahasa Inggris daring menurun dua bulan terakhir.",
        "Iklan tetap mendatangkan pengunjung ke halaman kelas, tetapi sebagian besar keluar saat diminta mengisi formulir panjang sepuluh kolom.",
        "Pengunjung yang melihat halaman kelas rajin mendaftar, tetapi jumlah kunjungan halaman turun drastis setelah jadwal unggah konten berhenti.",
        "Uji penyederhanaan formulir pendaftaran menjadi nama, kontak, dan jadwal pilihan. Ukur rasio penyelesaian formulir.",
        "Uji jadwal rutin tiga konten contoh pelajaran per minggu. Ukur jumlah kunjungan mingguan ke halaman kelas.",
        "Tanyakan apakah penurunan pendaftar disebabkan berkurangnya kunjungan halaman atau tingginya kegagalan di formulir daftar.",
    ),
    (
        "Tingkat kehadiran anggota pusat kebugaran merosot pada bulan kedua.",
        "Anggota baru bingung memakai alat beban karena sesi pengenalan awal jarang dijadwalkan pada pekan pertama.",
        "Anggota paham cara memakai seluruh alat, tetapi kelas malam yang paling diminati sering dibatalkan mendadak karena kekurangan pelatih.",
        "Uji jadwal sesi pengenalan alat wajib lima belas menit saat daftar ulang. Ukur kehadiran anggota pada pekan kelima.",
        "Uji daftar pelatih cadangan khusus kelas malam hari kerja. Ukur persentase kelas malam yang berjalan sesuai jadwal.",
        "Tanyakan apakah anggota berhenti datang karena bingung memakai alat atau karena kelas favorit sering batal.",
    ),
)

_CAUSAL_SCENARIOS_EN = (
    (
        "Dental clinic patient wait times keep rising every weekend.",
        "Queues pile up at reception because new patients still fill out paper intake forms. Treatment rooms sit empty waiting for charts.",
        "Reception check-in finishes in under three minutes. Delays happen because instrument sterilization between visits runs behind schedule.",
        "Trial pre-visit digital intake forms sent before arrival. Measure patient waiting time at reception.",
        "Trial adding one backup sterile tray set per treatment room. Measure turnover time between patients.",
        "Ask whether patients are waiting at reception check-in or waiting for treatment room turnaround.",
    ),
    (
        "Office catering deliveries frequently miss the lunch cutoff.",
        "The kitchen finishes packing on time at eleven. Drivers lose twenty minutes looking for loading docks and reception contacts.",
        "Drivers reach client lobbies quickly once loaded. Kitchen prep starts late because produce vendors often arrive past nine.",
        "Trial confirming loading dock details and lobby contact one day ahead. Measure on-site drop-off time.",
        "Trial shifting vendor produce delivery two hours earlier. Measure kitchen packing completion time.",
        "Ask whether delays happen during kitchen prep or during driver drop-off at client buildings.",
    ),
    (
        "Coworking space monthly memberships are churning faster this quarter.",
        "New members rarely return after week one because video calls drop in the meeting booths due to weak Wi-Fi.",
        "Wi-Fi speeds stay reliable all day. Members cancel because meeting rooms stay booked solid during peak afternoon hours.",
        "Trial a dedicated access point and backup router for meeting booths. Measure weekly Wi-Fi disconnect reports.",
        "Trial capping single-team room bookings to two peak hours per day. Measure afternoon room availability rate.",
        "Ask whether cancellations stem from Wi-Fi dropouts or from lack of meeting room availability.",
    ),
    (
        "Rewash requests at our commercial laundry shop doubled in two weeks.",
        "Most complaints cite collar stains missed because pre-wash stain sorting gets skipped during rush intake.",
        "Garments leave the washers spotless. Rewashes happen because steam irons drip mineral water marks onto dress shirts.",
        "Trial a mandatory collar-stain check tag before wash loading. Measure the percentage of stain-related rewashes.",
        "Trial descaling iron steam tanks every morning before opening. Measure water-mark complaints on pressed shirts.",
        "Ask whether rewashes come from missed pre-wash stains or from water marks added during pressing.",
    ),
    (
        "Corporate brochure print jobs keep needing costly reprints.",
        "Colors match the proof, but trim edges clip text because bleed margins on client files are not checked before printing.",
        "Trim margins are consistently accurate. Reprints happen because press color drifts after several hundred sheets without recalibration.",
        "Trial an automated bleed-margin preflight check before queuing jobs. Measure trim-error reprint rate.",
        "Trial press color calibration every five hundred sheets. Measure color-drift reprint rate.",
        "Ask whether reprints are caused by unchecked client bleed margins or by mid-run color drift on the press.",
    ),
    (
        "Weekday revenue at our car wash bay has slipped this month.",
        "Daily car count is unchanged, but far fewer drivers are choosing the interior vacuum and polish add-on.",
        "Arriving drivers still buy premium packages at the same rate. Total car traffic fell after sidewalk roadwork blocked the entrance sign.",
        "Trial a ten-minute express vacuum add-on at checkout. Measure add-on attachment rate per car.",
        "Trial temporary directional signage ahead of the roadwork detour. Measure daily weekday car entries.",
        "Ask whether revenue fell because fewer cars enter or because fewer drivers add interior packages.",
    ),
    (
        "Tile shipments from our building supply depot face rising returns.",
        "Orders contain the exact SKU requested, but corner tiles crack in transit because cartons lack corner cardboard guards.",
        "Shipments arrive unbroken. Returns happen because warehouse pickers confuse similar paint shade codes on printed slips.",
        "Trial cardboard corner protectors inside every tile carton. Measure transit-damage return rate.",
        "Trial barcode scanning each shade code before truck loading. Measure wrong-item return rate.",
        "Ask whether returns are mainly due to transit breakage or to wrong product codes being picked.",
    ),
    (
        "Hotel breakfast package attach rate is lower than forecast.",
        "Most room-only guests never hear about the discounted breakfast bundle during front-desk check-in.",
        "Guests know the breakfast bundle price, but skip it after seeing long buffet lines at seven in the morning.",
        "Trial a brief breakfast bundle prompt card at front-desk check-in. Measure breakfast attach rate per booking.",
        "Trial an express coffee and pastry counter from seven to eight. Measure average breakfast line wait time.",
        "Ask whether guests miss breakfast info at check-in or avoid the buffet because of morning queues.",
    ),
    (
        "Pharmacy shoppers often abandon baskets right at the checkout counter.",
        "Requested items are in stock, but checkout lines stall because prescription checks happen at the exact same register.",
        "Checkout lines move briskly. Shoppers abandon custom compounded orders because total price is only quoted after mixing.",
        "Trial separating prescription intake from standard retail checkout. Measure average checkout queue time.",
        "Trial providing a written price quote before compounding begins. Measure abandoned custom order rate.",
        "Ask whether customers leave due to checkout wait times or due to surprise pricing after mixing.",
    ),
    (
        "Cold-chain frozen deliveries to regional distributors keep getting rejected.",
        "Refrigerated trucks arrive on schedule, but cargo temperature rises past threshold because rear door seals leak warm air.",
        "Cargo stays below freezing on every trip. Rejections occur because trucks reach regional docks after receiving hours close.",
        "Trial replacing rear door rubber seals and logging temperature every two hours. Measure in-spec arrival rate.",
        "Trial departing ninety minutes earlier on outer regional routes. Measure on-time arrival before dock closing.",
        "Ask whether distributor rejections are caused by cargo temperature drift or by late arrival past dock hours.",
    ),
    (
        "Enrollments for our online coding workshop fell over two months.",
        "Traffic to the workshop page is steady, but most visitors drop off halfway through our ten-field registration form.",
        "Page visitors convert steadily, but total page visits plunged after weekly sample tutorial posts stopped.",
        "Trial trimming registration to name, email, and preferred cohort. Measure form completion rate.",
        "Trial publishing three short sample exercises each week. Measure weekly workshop page visits.",
        "Ask whether enrollments fell because page traffic dropped or because visitors abandon the registration form.",
    ),
    (
        "Gym member attendance drops sharply during the second month.",
        "New signups feel unsure using strength machines because onboarding walkthroughs rarely get booked in week one.",
        "Members use the equipment comfortably, but popular evening group classes get canceled last minute when instructors call out.",
        "Trial a fifteen-minute equipment walkthrough scheduled at signup. Measure member check-ins during week five.",
        "Trial maintaining a standby instructor roster for weekday evening classes. Measure evening class completion rate.",
        "Ask whether attendance falls because new members lack equipment guidance or because evening classes cancel.",
    ),
)


def _fmt_int_id(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _fmt_rp(n: int) -> str:
    return "Rp" + _fmt_int_id(n)


def iter_synth_business_midtrain(lang: str, target_docs: int, seed: int = 2026) -> Iterator[tuple[str, str | None]]:
    """Yield (document_text, mc_letter_or_None) for high-signal bilingual reasoning mid-training.

    Covers 6 core reasoning families:
      0. Multi-entity selective lookup (3-4 entities, any position, explicit step check)
      1. Min/max superlative comparison across 3-4 entities (explicit numeric ordering)
      2. 3-part / 4-part numeric breakdown (Total / Passed / Failed)
      3. Unanswerable metric-vs-headcount/volume contrast (explicit unit check + missing ask)
      4. Markdown table + exact step-by-step KPI math (pct_change fwd/rev, margin, conv, AOV, BE, ROI)
      5. Causal confirmed-fact diagnosis (free-form + 1:1:1 symmetric cyclic A/B/C MC)
    """
    rng = random.Random(seed + (101 if lang == "id" else 202))
    scenarios = _CAUSAL_SCENARIOS_ID if lang == "id" else _CAUSAL_SCENARIOS_EN
    cities = _CITIES_ID if lang == "id" else _CITIES_EN
    items = _ITEMS_ID if lang == "id" else _ITEMS_EN
    groups = _GROUPS_ID if lang == "id" else _GROUPS_EN
    months = _MONTHS_ID if lang == "id" else _MONTHS_EN
    days_id = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu")
    days_en = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")

    # Emit exact multiples of 6 so every family (and A/B/C letters in family 5) stays balanced
    n_cycles = max(1, target_docs // 6)
    mc_counter = 0

    for c in range(n_cycles):
        city = cities[c % len(cities)]
        month = months[(c // len(cities)) % len(months)]
        grp = groups[c % len(groups)]
        cycle_tag = (c // (len(cities) * len(months))) + 1

        # --- Family 0: Multi-entity selective lookup (3-4 entities, random target position) ---
        k_ent = 3 if c % 2 == 0 else 4
        ents = list(grp[:k_ent])
        vals = rng.sample(range(3, 65), k_ent)
        target_pos = c % k_ent
        t_ent, t_val = ents[target_pos], vals[target_pos]
        total_vol = rng.randint(450, 4800)
        if lang == "id":
            unit = ("menit", "hari", "jam")[c % 3]
            metric = ("waktu balas rata-rata", "lead time pengiriman", "waktu proses pesanan")[c % 3]
            ev_parts = ". ".join(f"{e}: {v} {unit}" for e, v in zip(ents, vals))
            doc0 = (
                f"User: Berapa {metric} untuk {t_ent} di {city} pada bulan {month} (periode {cycle_tag})?\n"
                f"Evidence:\n[1] Laporan Operasional {city} ({month}, periode {cycle_tag})\n"
                f"Total volume: {_fmt_int_id(total_vol)} pesanan. {ev_parts}.\n"
                f"Assistant: Berdasarkan Laporan Operasional {city} [1], angka masing-masing unit adalah {ev_parts}. "
                f"Untuk {t_ent}, {metric} adalah {t_val} {unit}."
            )
        else:
            unit = ("minutes", "days", "hours")[c % 3]
            metric = ("average response time", "delivery lead time", "order handling time")[c % 3]
            ev_parts = ". ".join(f"{e}: {v} {unit}" for e, v in zip(ents, vals))
            doc0 = (
                f"User: What was the {metric} reported for {t_ent} in {city} during {month} (batch {cycle_tag})?\n"
                f"Evidence:\n[1] {city} Operations Digest ({month}, batch {cycle_tag})\n"
                f"Total volume: {total_vol:,} orders. {ev_parts}.\n"
                f"Assistant: According to the {city} Operations Digest [1], the reported values are {ev_parts}. "
                f"Specifically for {t_ent}, the {metric} was {t_val} {unit}."
            )
        yield doc0, None

        # --- Family 1: Argmax / Argmin superlative comparison with explicit numeric sorting ---
        pick_items = rng.sample(items, 3)
        if c % 2 == 0:
            # 3 guaranteed-distinct positive shares summing cleanly to 100% (zero unbounded while-loops!)
            a = rng.randint(36, 56)
            b = rng.randint(22, min(35, 92 - a))
            d = 100 - a - b
            if d in (a, b):
                b -= 1
                d += 1
            shares = [a, b, d]
            rng.shuffle(shares)
            pairs = list(zip(pick_items, shares))
            want_max = (c // 2) % 2 == 0
            winner = max(pairs, key=lambda x: x[1]) if want_max else min(pairs, key=lambda x: x[1])
            ordered = sorted(pairs, key=lambda x: x[1], reverse=want_max)
            cmp_sym = ">" if want_max else "<"
            chain = f" {cmp_sym} ".join(f"{n} ({v}%)" for n, v in ordered)
            if lang == "id":
                sup = ("terbesar" if c % 4 == 0 else "tertinggi") if want_max else ("terkecil" if c % 4 == 2 else "terendah")
                ev = ", ".join(f"{n}: {v}% omzet" for n, v in pairs)
                doc1 = (
                    f"User: Produk mana yang menyumbang kontribusi omzet {sup} di cabang {city} bulan {month}?\n"
                    f"Evidence:\n[1] Ringkasan Penjualan Cabang {city} ({month})\n{ev}.\n"
                    f"Assistant: Urutan dari {sup} pada ringkasan [1]: {chain}. "
                    f"Jadi kontribusi omzet {sup} berasal dari {winner[0]} sebesar {winner[1]}%."
                )
            else:
                sup = ("largest" if c % 4 == 0 else "highest") if want_max else ("smallest" if c % 4 == 2 else "lowest")
                ev = ", ".join(f"{n}: {v}% of revenue" for n, v in pairs)
                doc1 = (
                    f"User: Which product contributed the {sup} share of revenue in {city} for {month}?\n"
                    f"Evidence:\n[1] {city} Sales Summary ({month})\n{ev}.\n"
                    f"Assistant: Ordering the figures in [1]: {chain}. "
                    f"Therefore, {winner[0]} had the {sup} revenue share at {winner[1]}%."
                )
        else:
            # 3 guaranteed-distinct 1-decimal percentages via rng.sample (O(1), zero while-loops!)
            accs = [round(v * 0.1, 1) for v in rng.sample(range(901, 996), 3)]
            ents3 = list(grp[:3])
            pairs = list(zip(ents3, accs))
            want_min = (c // 2) % 2 == 0
            winner = min(pairs, key=lambda x: x[1]) if want_min else max(pairs, key=lambda x: x[1])
            ordered = sorted(pairs, key=lambda x: x[1], reverse=not want_min)
            cmp_sym = "<" if want_min else ">"
            chain = f" {cmp_sym} ".join(f"{n} ({v}%)" for n, v in ordered)
            if lang == "id":
                sup = "terendah" if want_min else "tertinggi"
                ev = ". ".join(f"{n}: ketepatan {v}%" for n, v in pairs)
                doc1 = (
                    f"User: Unit mana yang mencatat tingkat ketepatan {sup} pada audit {city} bulan {month}?\n"
                    f"Evidence:\n[1] Audit Operasional {city} ({month})\n{ev}.\n"
                    f"Assistant: Perbandingan angka di audit [1]: {chain}. "
                    f"Unit dengan tingkat ketepatan {sup} adalah {winner[0]} sebesar {winner[1]}%."
                )
            else:
                sup = "lowest" if want_min else "highest"
                ev = ". ".join(f"{n}: {v}% accuracy" for n, v in pairs)
                doc1 = (
                    f"User: Which unit had the {sup} fulfillment accuracy in the {city} audit ({month})?\n"
                    f"Evidence:\n[1] {city} Audit Report ({month})\n{ev}.\n"
                    f"Assistant: Comparing the units in [1]: {chain}. "
                    f"{winner[0]} recorded the {sup} accuracy at {winner[1]}%."
                )
        yield doc1, None

        # --- Family 2: 3-part numeric breakdown (Total / Passed / Failed) ---
        tot = rng.randint(600, 3200)
        failed = rng.randint(14, 145)
        passed = tot - failed
        if lang == "id":
            dom = (
                ("tiket layanan", "selesai dalam 24 jam", "melewati batas SLA 24 jam"),
                ("paket kiriman", "tiba tepat waktu", "diretur karena alamat keliru"),
                ("faktur tagihan", "dibayar tepat waktu", "melewati jatuh tempo"),
                ("pesanan harian", "terkirim lengkap", "dibatalkan sebelum kirim"),
            )[c % 4]
            noun, ok_lbl, bad_lbl = dom
            ask_bad = (c % 3 != 0)
            q_part = f"Berapa jumlah {noun} yang {bad_lbl}" if ask_bad else f"Berapa jumlah {noun} yang {ok_lbl}"
            ans_n = failed if ask_bad else passed
            ans_lbl = bad_lbl if ask_bad else ok_lbl
            doc2 = (
                f"User: {q_part} di {city} pada bulan {month}?\n"
                f"Evidence:\n[1] Rekap {city} ({month})\n"
                f"Total {noun}: {_fmt_int_id(tot)}. {ok_lbl.capitalize()}: {_fmt_int_id(passed)}. "
                f"{bad_lbl.capitalize()}: {_fmt_int_id(failed)} {noun}.\n"
                f"Assistant: Berdasarkan Rekap {city} [1], dari total {_fmt_int_id(tot)} {noun}, "
                f"jumlah yang {ans_lbl} adalah {_fmt_int_id(ans_n)} {noun}."
            )
        else:
            dom = (
                ("support tickets", "resolved within 24 hours", "breached the 24-hour SLA"),
                ("outbound parcels", "delivered on schedule", "returned due to address error"),
                ("client invoices", "paid before due date", "became overdue"),
                ("production batches", "passed quality inspection", "failed quality inspection"),
            )[c % 4]
            noun, ok_lbl, bad_lbl = dom
            ask_bad = (c % 3 != 0)
            q_part = f"How many {noun} {bad_lbl}" if ask_bad else f"How many {noun} {ok_lbl}"
            ans_n = failed if ask_bad else passed
            ans_lbl = bad_lbl if ask_bad else ok_lbl
            doc2 = (
                f"User: {q_part} in {city} during {month}?\n"
                f"Evidence:\n[1] {city} Monthly Summary ({month})\n"
                f"Total {noun}: {tot:,}. {ok_lbl.capitalize()}: {passed:,}. {bad_lbl.capitalize()}: {failed:,} {noun}.\n"
                f"Assistant: According to the {city} Monthly Summary [1], out of {tot:,} total {noun}, {ans_n:,} {noun} {ans_lbl}."
            )
        yield doc2, None

        # --- Family 3: Unanswerable metric-vs-headcount/volume contrast ---
        n_ppl = rng.randint(120, 6800)
        n_tx = rng.randint(85, 2400)
        pct_att = rng.randint(84, 99)
        if lang == "id":
            utopic = (
                (f"Berapa laba bersih cabang {city} bulan {month}?",
                 f"Total pengunjung toko: {_fmt_int_id(n_ppl)} orang, dengan {_fmt_int_id(n_tx)} transaksi selesai.",
                 "laba bersih", "jumlah pengunjung (orang) dan transaksi, bukan data keuangan laba bersih",
                 "total omzet dan total biaya operasional"),
                (f"Berapa biaya sewa lokasi operasional di {city} per bulan?",
                 f"Jumlah staf aktif: {max(12, n_ppl // 100)} orang dengan tingkat kehadiran {pct_att}%.",
                 "biaya sewa lokasi", "jumlah staf dan persentase kehadiran, bukan nominal biaya sewa",
                 "dokumen kontrak atau rincian biaya sewa"),
                (f"Mengapa pembatalan pesanan di {city} naik pada bulan {month}?",
                 f"Total pesanan tercatat: {_fmt_int_id(n_tx)} transaksi, berubah {rng.randint(15, 90)} transaksi dibanding bulan lalu.",
                 "penyebab kenaikan pembatalan pesanan", "volume total pesanan tanpa alasan pembatalan",
                 "log alasan pembatalan pelanggan"),
                (f"Berapa persen diskon rata-rata yang diberikan tim penjualan di {city}?",
                 f"Jumlah mitra aktif: {rng.randint(25, 190)} toko, termasuk {rng.randint(3, 18)} mitra baru.",
                 "persen diskon rata-rata", "jumlah mitra aktif tanpa catatan potongan harga",
                 "rekap diskon atau faktur penjualan"),
            )[c % 4]
            uq, uev, umiss, uwhy, uask = utopic
            doc3 = (
                f"User: {uq}\n"
                f"Evidence:\n[1] Laporan Aktivitas {city} ({month})\n{uev}\n"
                f"Assistant: Kutipan [1] tidak menyebutkan {umiss} karena hanya memuat {uwhy}. "
                f"Bisa bagikan {uask} yang terkait?"
            )
        else:
            utopic = (
                (f"What was the net profit margin of the {city} branch in {month}?",
                 f"Store foot traffic reached {n_ppl:,} visitors across {n_tx:,} completed orders.",
                 "the net profit margin", "visitor and order counts rather than profit or cost figures",
                 "revenue, cost of goods sold, and operating expenses"),
                (f"What was the monthly warehouse lease cost in {city}?",
                 f"Active headcount: {max(12, n_ppl // 100)} employees with {pct_att}% attendance this month.",
                 "the monthly lease cost", "headcount and attendance rate rather than facility expenses",
                 "the lease agreement or expense ledger"),
                (f"Why did customer refunds increase in {city} during {month}?",
                 f"Total completed orders: {n_tx:,}, shifting by {rng.randint(15, 95)} orders from the prior month.",
                 "the root cause of higher refunds", "order volume totals without refund reason codes",
                 "the breakdown of customer refund reasons"),
                (f"What was the exact customer acquisition cost (CAC) for {city} in {month}?",
                 f"Active accounts totaled {rng.randint(30, 240)} businesses, including {rng.randint(4, 22)} new signups.",
                 "the customer acquisition cost", "account counts without marketing or sales spend",
                 "the marketing spend for the same period"),
            )[c % 4]
            uq, uev, umiss, uwhy, uask = utopic
            doc3 = (
                f"User: {uq}\n"
                f"Evidence:\n[1] {city} Activity Report ({month})\n{uev}\n"
                f"Assistant: The provided excerpt [1] does not specify {umiss} because it only reports {uwhy}. "
                f"Could you share {uask}?"
            )
        yield doc3, None

        # --- Family 4: Markdown Table + Exact Step-by-Step KPI Derivation ---
        kpi_kind = c % 4
        if kpi_kind == 0:
            # Multiples of 20 so every 5% step gives an EXACT integer new_v (zero rounding mismatch!)
            base_v = rng.choice((20, 40, 60, 80, 100, 120, 160, 200))
            delta_pct = rng.choice((-30, -25, -20, -15, -10, 10, 15, 20, 25, 30, 40))
            assert (base_v * delta_pct) % 100 == 0
            new_v = base_v + (base_v * delta_pct) // 100
            o1, o2 = rng.randint(80, 400), rng.randint(70, 420)
            rev_dir = (c % 8 == 4)
            if rev_dir:
                p_from, p_to, v_from, v_to = ("M2", "M1", new_v, base_v) if lang == "id" else ("W2", "W1", new_v, base_v)
            else:
                p_from, p_to, v_from, v_to = ("M1", "M2", base_v, new_v) if lang == "id" else ("W1", "W2", base_v, new_v)
            actual_pct = round((v_to - v_from) / v_from * 100.0, 2)
            pct_s = f"{int(actual_pct)}" if actual_pct == int(actual_pct) else f"{actual_pct:.2f}"
            if lang == "id":
                r1, r2 = base_v * 1_000_000, new_v * 1_000_000
                doc4 = (
                    f"User: Rekap cabang {city} ({month}):\n| Minggu | Omzet | Pesanan |\n|---|---|---|\n"
                    f"| M1 | {_fmt_rp(r1)} | {o1} |\n| M2 | {_fmt_rp(r2)} | {o2} |\n"
                    f"Berapa persen perubahan omzet dari {p_from} ke {p_to}?\n"
                    f"Assistant: Rumus perubahan persentase: ({v_to} juta - {v_from} juta) / {v_from} juta x 100% = {pct_s}%. "
                    f"Perubahan omzet dari {p_from} ke {p_to}: {pct_s}%."
                )
            else:
                r1, r2 = base_v * 1000, new_v * 1000
                doc4 = (
                    f"User: {city} summary ({month}):\n| Week | Revenue | Orders |\n|---|---|---|\n"
                    f"| W1 | ${r1:,} | {o1} |\n| W2 | ${r2:,} | {o2} |\n"
                    f"What is the revenue percentage change from {p_from} to {p_to}?\n"
                    f"Assistant: Percentage change formula: (${v_to * 1000:,} - ${v_from * 1000:,}) / ${v_from * 1000:,} * 100 = {pct_s}%. "
                    f"Revenue change from {p_from} to {p_to}: {pct_s}%."
                )
        elif kpi_kind == 1:
            rev_u = rng.choice((20, 40, 50, 60, 80, 100, 150, 200))
            gm_pct = rng.choice((20, 25, 30, 35, 40, 45, 50, 60))
            cogs_u = int(round(rev_u * (100 - gm_pct) / 100))
            gp_u = rev_u - cogs_u
            if lang == "id":
                doc4 = (
                    f"User: Omzet {city} bulan {month} Rp{rev_u} juta dan HPP Rp{cogs_u} juta. Berapa margin kotornya?\n"
                    f"Assistant: Laba kotor = Rp{rev_u} juta - Rp{cogs_u} juta = Rp{gp_u} juta. "
                    f"Margin kotor = ({gp_u} / {rev_u}) x 100% = {gm_pct}%."
                )
            else:
                doc4 = (
                    f"User: {city} monthly revenue was ${rev_u * 1000:,} with COGS of ${cogs_u * 1000:,}. What is the gross margin?\n"
                    f"Assistant: Gross profit = ${rev_u * 1000:,} - ${cogs_u * 1000:,} = ${gp_u * 1000:,}. "
                    f"Gross margin = ({gp_u * 1000:,} / {rev_u * 1000:,}) * 100 = {gm_pct}%."
                )
        elif kpi_kind == 2:
            vis = rng.choice((200, 400, 500, 800, 1000, 2000, 2500))
            cr_pct = rng.choice((2, 4, 5, 6, 8, 10, 12, 15))
            buyers = vis * cr_pct // 100
            if lang == "id":
                doc4 = (
                    f"User: Dari {_fmt_int_id(vis)} pengunjung toko di {city} ({month}), ada {_fmt_int_id(buyers)} pembeli. "
                    f"Berapa tingkat konversinya?\n"
                    f"Assistant: Tingkat konversi = ({_fmt_int_id(buyers)} / {_fmt_int_id(vis)}) x 100% = {cr_pct}%."
                )
            else:
                doc4 = (
                    f"User: Out of {vis:,} visitors at the {city} store ({month}), {buyers:,} placed an order. "
                    f"What is the conversion rate?\n"
                    f"Assistant: Conversion rate = ({buyers:,} / {vis:,}) * 100 = {cr_pct}%."
                )
        else:
            cm = rng.choice((10, 15, 20, 25, 40, 50))
            be_units = rng.choice((100, 150, 200, 250, 300, 400, 600, 800))
            fixed = cm * be_units
            var_c = rng.choice((15, 20, 30, 40, 50))
            price = var_c + cm
            if lang == "id":
                doc4 = (
                    f"User: Biaya tetap bulanan cabang {city} {_fmt_rp(fixed * 1000)}, harga jual {_fmt_rp(price * 1000)} per unit, "
                    f"biaya variabel {_fmt_rp(var_c * 1000)} per unit. Berapa titik impasnya?\n"
                    f"Assistant: Margin kontribusi = {_fmt_rp(price * 1000)} - {_fmt_rp(var_c * 1000)} = {_fmt_rp(cm * 1000)} per unit. "
                    f"Titik impas = {_fmt_rp(fixed * 1000)} / {_fmt_rp(cm * 1000)} = {be_units} unit."
                )
            else:
                doc4 = (
                    f"User: Monthly fixed cost in {city} is ${fixed:,}, unit price is ${price}, and unit variable cost is ${var_c}. "
                    f"What is the break-even volume?\n"
                    f"Assistant: Contribution margin = ${price} - ${var_c} = ${cm} per unit. "
                    f"Break-even volume = ${fixed:,} / ${cm} = {be_units} units."
                )
        yield doc4, None

        # --- Family 5: Causal Confirmed-Fact Action Selection (1:1:1 symmetric A/B/C MC + free-form) ---
        sc = scenarios[c % len(scenarios)]
        issue, fa, fb, act_a, act_b, ask_q = sc
        choices = (act_a, act_b, ask_q)
        fact_i = c % 3
        day_lbl = days_id[c % len(days_id)] if lang == "id" else days_en[c % len(days_en)]
        wk_num = (c // len(scenarios)) % 52 + 1
        if lang == "id":
            hdr = "Fakta terkonfirmasi"
            instr = "Pilih satu tindakan dan ukuran yang cocok. Jika fakta kurang, pilih pertanyaan."
            missing_txt = "Belum diketahui di bagian mana proses atau pelanggan tertahan."
            note = f"Catatan cabang {city} minggu ke-{wk_num} dicatat pada hari {day_lbl} ({month})."
        else:
            hdr = "Confirmed facts"
            instr = "Choose one action with its metric. If a fact is missing, choose the question."
            missing_txt = "It is not yet known where customers or operations are getting held up."
            note = f"{city} branch note for week {wk_num} was logged on {day_lbl} ({month})."
        confirmed = (fa, fb, missing_txt)[fact_i]
        body = f"{issue}\n{hdr}: {confirmed} {note}\n{instr}"

        # Alternate MC letter doc (exact 1:1:1 A/B/C!) and free-form causal explanation
        if c % 2 == 0:
            target_slot = mc_counter % 3
            mc_counter += 1
            offset = (fact_i - target_slot) % 3
            order = [(offset + k) % 3 for k in range(3)]
            opts_txt = "\n".join(f"{ltr}. {choices[idx_c]}" for ltr, idx_c in zip("ABC", order))
            gold_letter = "ABC"[order.index(fact_i)]
            assert gold_letter == "ABC"[target_slot]
            doc5 = f"User: {body}\n{opts_txt}\nAssistant: {gold_letter}"
            yield doc5, gold_letter
        else:
            doc5 = f"User: {body}\nAssistant: {choices[fact_i]}"
            yield doc5, None


# ---------------------------------------------------------------------------
# Multiprocessing tokenizer pool + streaming packer with zero-leakage guard
# ---------------------------------------------------------------------------

_WORKER_TOK: BPETokenizer | None = None


def _worker_init(tok_path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = BPETokenizer.load(tok_path)


def _worker_encode_batch(texts: list[str]) -> list[list[int]]:
    assert _WORKER_TOK is not None
    return [_WORKER_TOK.encode(t, add_eos=True) for t in texts]


def pack_language_stream(
    sources_with_targets: list[tuple[str, int, Iterator[str]]],
    total_train_target: int,
    val_target: int,
    out_dir: Path,
    lang: str,
    tokenizer_path: Path,
    eval_pattern: re.Pattern[str] | None,
    pool: mp.pool.Pool | None,
    batch_docs: int = 1024,
) -> dict:
    """Pack multiple sources into train.{lang}.bin + val.{lang}.bin with strict guards.

    Unused token cap from finite datasets automatically carries over into the final
    replay anchor source so total train tokens reach `total_train_target` cleanly
    without ever over-repeating small datasets.
    """
    train_writer = BinWriter(out_dir / f"train.{lang}.bin")
    val_writer = BinWriter(out_dir / f"val.{lang}.bin")
    seen: set[bytes] = set()
    per_source: dict[str, dict[str, int]] = {}
    blocked_eval_hits = 0
    local_tok = None if pool is not None else BPETokenizer.load(tokenizer_path)

    val_per_src = max(1, val_target // max(1, len(sources_with_targets)))

    def encode_batch(batch: list[str]) -> list[list[int]]:
        if pool is not None:
            chunk = max(1, len(batch) // max(1, pool._processes))
            sub = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]
            out: list[list[int]] = []
            for part in pool.map(_worker_encode_batch, sub):
                out.extend(part)
            return out
        assert local_tok is not None
        return [local_tok.encode(t, add_eos=True) for t in batch]

    try:
        for idx_src, (src_key, base_cap, iterator) in enumerate(sources_with_targets):
            is_last = idx_src == len(sources_with_targets) - 1
            eff_cap = max(1, total_train_target - train_writer.n_tokens) if is_last else base_cap
            src_val_cap = max(val_per_src, val_target - val_writer.n_tokens) if is_last else val_per_src

            src_train = 0
            src_val = 0
            src_docs = 0
            src_blocked = 0
            pending_train: list[str] = []
            pending_val: list[str] = []
            t0 = time.time()

            def flush_buffers() -> None:
                nonlocal src_train, src_val
                if pending_val:
                    for ids in encode_batch(pending_val):
                        if src_val < src_val_cap:
                            val_writer.write(ids)
                            src_val += len(ids)
                    pending_val.clear()
                if pending_train:
                    for ids in encode_batch(pending_train):
                        if src_train < eff_cap:
                            train_writer.write(ids)
                            src_train += len(ids)
                    pending_train.clear()

            for text in iterator:
                if is_eval_contaminated(text, eval_pattern):
                    blocked_eval_hits += 1
                    src_blocked += 1
                    continue
                digest = blake2b(text.encode("utf-8"), digest_size=16).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                src_docs += 1

                is_val = int.from_bytes(digest[:2], "big") % 23 == 0
                if is_val:
                    if src_val < src_val_cap:
                        pending_val.append(text)
                else:
                    if src_train < eff_cap:
                        pending_train.append(text)

                if (
                    len(pending_train) + len(pending_val) >= batch_docs
                    or (src_train >= eff_cap and len(pending_val) >= 8)
                ):
                    flush_buffers()
                    if src_train >= eff_cap and src_val >= src_val_cap:
                        break

            flush_buffers()
            per_source[src_key] = {
                "train": src_train,
                "val": src_val,
                "docs": src_docs,
                "blocked_eval": src_blocked,
            }
            elapsed = time.time() - t0
            print(
                f"  [midtrain:{lang}] {src_key:22s}: {src_train:11,} train | {src_val:7,} val "
                f"| {src_docs:7,} docs | blocked={src_blocked} | {elapsed:.1f}s",
                flush=True,
            )
    finally:
        train_writer.close()
        val_writer.close()

    return {
        "train": train_writer.n_tokens,
        "val": val_writer.n_tokens,
        "blocked_eval_hits": blocked_eval_hits,
        "by_source": per_source,
    }


def _wrap_synth(lang: str, target_docs: int, letter_counts: Counter) -> Iterator[str]:
    for text, letter in iter_synth_business_midtrain(lang, target_docs):
        if letter is not None:
            letter_counts[letter] += 1
        yield text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/kuon1-bilingual-24k.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--val-tokens", type=int, default=120_000, help="per language")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--offline", action="store_true", help="pack synthetic bilingual corpus only (for tests)")
    ap.add_argument("--allow-custom-tokenizer", action="store_true", help="skip exact SHA-256 lock (tests only)")
    args = ap.parse_args()

    if not args.tokenizer.is_file():
        ap.error(f"Tokenizer not found: {args.tokenizer}")
    tok_sha = sha256(args.tokenizer.read_bytes()).hexdigest()
    if not args.allow_custom_tokenizer:
        assert tok_sha == EXPECTED_TOKENIZER_SHA256, (
            f"Tokenizer SHA-256 mismatch! got={tok_sha} expected={EXPECTED_TOKENIZER_SHA256}"
        )

    tokenizer = BPETokenizer.load(args.tokenizer)
    assert tokenizer.vocab_size <= 65535, "vocab_size exceeds uint16 range"
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.tokenizer, args.out / "tokenizer.json")

    fingerprints = build_forbidden_eval_fingerprints()
    eval_pattern = build_forbidden_eval_regex(fingerprints)
    print(
        f"[midtrain] Tokenizer verified (vocab={tokenizer.vocab_size}, sha256={tok_sha[:12]}...) "
        f"| eval_guards={len(fingerprints)} | workers={args.workers}",
        flush=True,
    )

    en_total = args.train_tokens * 60 // 100
    id_total = args.train_tokens - en_total

    en_mc_counts: Counter = Counter()
    id_mc_counts: Counter = Counter()

    if args.offline:
        en_docs = max(120, en_total // 45)
        id_docs = max(120, id_total // 45)
        en_sources = [("synth_biz_midtrain_en", en_total, _wrap_synth("en", en_docs, en_mc_counts))]
        id_sources = [("synth_biz_midtrain_id", id_total, _wrap_synth("id", id_docs, id_mc_counts))]
    else:
        # Scale proportions cleanly for any --train-tokens (1B default -> 600M EN + 400M ID)
        # Keep synthetic share tight at 2.2% of total corpus (10M EN + 12M ID = 22M) so natural
        # diverse reasoning/QA/math/instruction text stays 97.8% dominant without template overfitting.
        scale_en = en_total / 600_000_000
        scale_id = id_total / 400_000_000

        en_synth_cap = max(1, int(10_000_000 * scale_en))
        en_mathqa_cap = max(1, int(4_000_000 * scale_en))
        en_squad_cap = max(1, int(24_000_000 * scale_en))
        en_drop_cap = max(1, int(18_000_000 * scale_en))
        en_hotpot_cap = max(1, int(24_000_000 * scale_en))
        en_pleias_cap = max(1, int(155_000_000 * scale_en))
        en_smoltalk_cap = max(1, int(125_000_000 * scale_en))
        en_math_cap = max(1, int(100_000_000 * scale_en))
        en_cosmo_cap = max(1, int(70_000_000 * scale_en))
        en_fw_cap = max(
            1,
            en_total
            - en_synth_cap
            - en_mathqa_cap
            - en_squad_cap
            - en_drop_cap
            - en_hotpot_cap
            - en_pleias_cap
            - en_smoltalk_cap
            - en_math_cap
            - en_cosmo_cap,
        )

        id_tydi_cap = max(1, int(4_000_000 * scale_id))
        id_synth_cap = max(1, int(12_000_000 * scale_id))
        id_aya_cap = max(1, int(115_000_000 * scale_id))
        id_wiki_cap = max(1, int(90_000_000 * scale_id))
        id_fw_cap = max(1, id_total - id_tydi_cap - id_synth_cap - id_aya_cap - id_wiki_cap)

        # Oversize doc count generously (~100 tokens/doc avg) so synthetic stream always reaches its cap
        en_synth_docs = max(600, (en_synth_cap // 55) + 6000)
        id_synth_docs = max(600, (id_synth_cap // 55) + 6000)

        def iter_pleias_disjoint() -> Iterator[str]:
            for s in range(20, 120):
                yield from _iter_pleias_synth_fast(s)

        def iter_cosmo_disjoint() -> Iterator[str]:
            for s in range(25, 104):
                yield from _iter_parquet_texts_fast(hf_shard_url("cosmopedia_v2", s))

        def iter_fw_edu_disjoint() -> Iterator[str]:
            for s in range(25, 234):
                yield from _iter_parquet_texts_fast(hf_shard_url("fineweb_edu", s))

        def iter_fw_id_disjoint() -> Iterator[str]:
            for s in range(20, 120):
                yield from _iter_parquet_texts_fast(hf_shard_url("fineweb2_hq_id", s), drop_noise_re=_FW_ID_NOISE_RE)

        en_sources = [
            ("synth_biz_midtrain_en", en_synth_cap, _wrap_synth("en", en_synth_docs, en_mc_counts)),
            ("finqa_gsm8k_en", en_mathqa_cap, iter_finqa_gsm8k_en()),
            ("squad_v2_en", en_squad_cap, iter_squad_v2_en()),
            ("drop_en", en_drop_cap, iter_drop_en()),
            ("hotpotqa_en", en_hotpot_cap, iter_hotpotqa_en()),
            ("pleias_synth_en", en_pleias_cap, iter_pleias_disjoint()),
            ("smoltalk_en", en_smoltalk_cap, iter_smoltalk_en()),
            ("finemath_combo_en", en_math_cap, iter_finemath_combo_en(finemath_start=15, infi_start=0)),
            ("cosmopedia_v2", en_cosmo_cap, iter_cosmo_disjoint()),
            ("fineweb_edu", en_fw_cap, iter_fw_edu_disjoint()),
        ]
        id_sources = [
            ("tydiqa_id", id_tydi_cap, iter_tydiqa_id()),
            ("synth_biz_midtrain_id", id_synth_cap, _wrap_synth("id", id_synth_docs, id_mc_counts)),
            ("aya_id_clean", id_aya_cap, iter_aya_id_clean()),
            ("wiki_id", id_wiki_cap, iter_wiki_id()),
            ("fineweb2_hq_id", id_fw_cap, iter_fw_id_disjoint()),
        ]

    t_all = time.time()
    pool = (
        mp.Pool(processes=args.workers, initializer=_worker_init, initargs=(str(args.tokenizer),))
        if args.workers > 1
        else None
    )
    try:
        print(f"[midtrain] Packing {en_total:,} EN + {id_total:,} ID train tokens -> {args.out}", flush=True)
        en_stats = pack_language_stream(
            en_sources, en_total, args.val_tokens, args.out, "en", args.tokenizer, eval_pattern, pool
        )
        id_stats = pack_language_stream(
            id_sources, id_total, args.val_tokens, args.out, "id", args.tokenizer, eval_pattern, pool
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Verify near-exact A/B/C symmetry on generated causal MC items (max diff <= 1 due to truncation at cap)
    for tag, cnt in (("en", en_mc_counts), ("id", id_mc_counts)):
        if cnt:
            assert set(cnt.keys()) == {"A", "B", "C"}, f"{tag} missing MC letters: {dict(cnt)}"
            assert max(cnt.values()) - min(cnt.values()) <= 2, f"{tag} skewed MC letters: {dict(cnt)}"

    meta = {
        "corpus": "kuon1-phase1.5-midtrain-1b",
        "vocab_size": tokenizer.vocab_size,
        "language_mix": {"en": 60, "id": 40},
        "tokenizer_sha256": tok_sha,
        "offline": bool(args.offline),
        "disjoint_shard_offsets": {
            "pleias_synth_en": 20,
            "finemath_4plus": 15,
            "infiwebmath_4plus": 0,
            "cosmopedia_v2": 25,
            "fineweb_edu": 25,
            "fineweb2_hq_id": 20,
        },
        "eval_fingerprints_checked": len(fingerprints),
        "synth_mc_letter_counts": {"en": dict(en_mc_counts), "id": dict(id_mc_counts)},
        "tokens": {"en": en_stats, "id": id_stats},
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[midtrain] DONE in {meta['elapsed_sec']}s | "
        f"EN {en_stats['train']:,} train / {en_stats['val']:,} val | "
        f"ID {id_stats['train']:,} train / {id_stats['val']:,} val | "
        f"mc_en={dict(en_mc_counts)} mc_id={dict(id_mc_counts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

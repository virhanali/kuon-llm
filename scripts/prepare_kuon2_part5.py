"""Pack Kuon-2 Reserve Part 5: 1.50B completely disjoint bilingual EN/ID pretrain tokens.

Zero shard overlap with all previous corpora:
  - Parts 1-3 (1.40B pretrain): shards 0..12
  - Midtrain  (1.00B):          pleias 20..25, finemath 15..25, infiwebmath 0..10,
                                cosmo 25..30, fw_edu 25..35, fw_id 20..35
  - Part 4    (1.00B):          pleias 60..70, finemath 42..55, cosmo 55..65,
                                fw_edu 80..105, fw_id 50..78
  - Part 5    (1.50B, here):    pleias >= 150, infiwebmath >= 16, cosmo >= 76,
                                fw_edu >= 140, fw_id >= 85 (spam-filtered).

Used as an optional Chinchilla-optimal extension (pushing Kuon-2 300M from 3.40B -> 4.90B
tokens) if the 3.40B Kuon-2.0 stop-gate benefits from extra foundation capacity.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Iterator

from kuon.tokenizer import BPETokenizer
from scripts.prepare_kuon1_corpus import hf_shard_url
from scripts.prepare_kuon1_midtrain import (
    EXPECTED_TOKENIZER_SHA256,
    _FW_ID_NOISE_RE,
    _iter_parquet_texts_fast,
    _iter_pleias_synth_fast,
    _worker_init,
    build_forbidden_eval_fingerprints,
    build_forbidden_eval_regex,
    pack_language_stream,
)

ROOT = Path(__file__).resolve().parents[1]

INFI_REV = "e92b25a616738fe95dc186b64dfb19f9c8525594"

OFFSETS = {
    "pleias_synth_en": (150, 500),
    "infiwebmath_4plus": (16, 32),
    "cosmopedia_v2": (76, 104),
    "fineweb_edu": (140, 234),
    "fineweb2_hq_id": (85, 120),
}


def iter_pleias_part5() -> Iterator[str]:
    lo, hi = OFFSETS["pleias_synth_en"]
    for s in range(lo, hi):
        yield from _iter_pleias_synth_fast(s)


def iter_infiwebmath_part5() -> Iterator[str]:
    lo, hi = OFFSETS["infiwebmath_4plus"]
    for s in range(lo, hi):
        url = (
            f"https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/{INFI_REV}/"
            f"infiwebmath-4plus/train-{s:05d}-of-00032.parquet"
        )
        yield from _iter_parquet_texts_fast(url)


def iter_cosmo_part5() -> Iterator[str]:
    lo, hi = OFFSETS["cosmopedia_v2"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("cosmopedia_v2", s))


def iter_fw_edu_part5() -> Iterator[str]:
    lo, hi = OFFSETS["fineweb_edu"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("fineweb_edu", s))


def iter_fw_id_part5() -> Iterator[str]:
    lo, hi = OFFSETS["fineweb2_hq_id"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("fineweb2_hq_id", s), drop_noise_re=_FW_ID_NOISE_RE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/kuon1-bilingual-24k.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-tokens", type=int, default=1_500_000_000)
    ap.add_argument("--val-tokens", type=int, default=120_000, help="per language")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = ap.parse_args()

    if not args.tokenizer.is_file():
        ap.error(f"Tokenizer not found: {args.tokenizer}")
    tok_sha = sha256(args.tokenizer.read_bytes()).hexdigest()
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
        f"[part5] Tokenizer verified (vocab={tokenizer.vocab_size}, sha256={tok_sha[:12]}...) "
        f"| eval_guards={len(fingerprints)} | workers={args.workers}",
        flush=True,
    )

    en_total = args.train_tokens * 60 // 100
    id_total = args.train_tokens - en_total

    en_sources = [
        ("pleias_synth_en", en_total * 20 // 100, iter_pleias_part5()),
        ("infiwebmath_4plus", en_total * 15 // 100, iter_infiwebmath_part5()),
        ("cosmopedia_v2", en_total * 25 // 100, iter_cosmo_part5()),
        ("fineweb_edu", en_total - (en_total * 60 // 100), iter_fw_edu_part5()),
    ]
    id_sources = [
        ("fineweb2_hq_id", id_total, iter_fw_id_part5()),
    ]

    t0 = time.time()
    pool = (
        mp.Pool(processes=args.workers, initializer=_worker_init, initargs=(str(args.tokenizer),))
        if args.workers > 1
        else None
    )
    try:
        print(f"[part5] Packing {en_total:,} EN + {id_total:,} ID train tokens -> {args.out}", flush=True)
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

    meta = {
        "corpus": "kuon2-pretrain-1.5b-part5",
        "vocab_size": tokenizer.vocab_size,
        "language_mix": {"en": 60, "id": 40},
        "tokenizer_sha256": tok_sha,
        "disjoint_shard_ranges": {k: [lo, hi] for k, (lo, hi) in OFFSETS.items()},
        "eval_fingerprints_checked": len(fingerprints),
        "tokens": {"en": en_stats, "id": id_stats},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[part5] DONE in {meta['elapsed_sec']}s | "
        f"EN {en_stats['train']:,} train / {en_stats['val']:,} val | "
        f"ID {id_stats['train']:,} train / {id_stats['val']:,} val",
        flush=True,
    )


if __name__ == "__main__":
    main()

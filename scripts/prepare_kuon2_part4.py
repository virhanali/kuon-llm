"""Pack Kuon-2 Stage 1B Part 4: 1.00B completely disjoint bilingual EN/ID pretrain tokens.

Zero shard overlap with all previous corpora:
  - Parts 1-3 (1.40B pretrain): shards 0..12
  - Midtrain  (1.00B):          shards 15..35
  - Part 4    (1.00B, here):    pleias_synth >= 60, finemath_4plus >= 42,
                                cosmopedia_v2 >= 55, fineweb_edu >= 80,
                                fineweb2_hq_id >= 50 (with Indonesian spam filter).

Reuses the exact Kuon 24,576-vocab tokenizer (SHA-256 locked), eval contamination
guard, and 6-worker multiprocessing packer from prepare_kuon1_midtrain.py.
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

OFFSETS = {
    "pleias_synth_en": (60, 500),
    "finemath_4plus": (42, 64),
    "cosmopedia_v2": (55, 104),
    "fineweb_edu": (80, 234),
    "fineweb2_hq_id": (50, 120),
}


def iter_pleias_part4() -> Iterator[str]:
    lo, hi = OFFSETS["pleias_synth_en"]
    for s in range(lo, hi):
        yield from _iter_pleias_synth_fast(s)


def iter_finemath_part4() -> Iterator[str]:
    lo, hi = OFFSETS["finemath_4plus"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("finemath_4plus", s))


def iter_cosmo_part4() -> Iterator[str]:
    lo, hi = OFFSETS["cosmopedia_v2"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("cosmopedia_v2", s))


def iter_fw_edu_part4() -> Iterator[str]:
    lo, hi = OFFSETS["fineweb_edu"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("fineweb_edu", s))


def iter_fw_id_part4() -> Iterator[str]:
    lo, hi = OFFSETS["fineweb2_hq_id"]
    for s in range(lo, hi):
        yield from _iter_parquet_texts_fast(hf_shard_url("fineweb2_hq_id", s), drop_noise_re=_FW_ID_NOISE_RE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/kuon1-bilingual-24k.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-tokens", type=int, default=1_000_000_000)
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
        f"[part4] Tokenizer verified (vocab={tokenizer.vocab_size}, sha256={tok_sha[:12]}...) "
        f"| eval_guards={len(fingerprints)} | workers={args.workers}",
        flush=True,
    )

    en_total = args.train_tokens * 60 // 100
    id_total = args.train_tokens - en_total

    en_sources = [
        ("pleias_synth_en", en_total * 15 // 100, iter_pleias_part4()),
        ("finemath_4plus", en_total * 15 // 100, iter_finemath_part4()),
        ("cosmopedia_v2", en_total * 25 // 100, iter_cosmo_part4()),
        ("fineweb_edu", en_total - (en_total * 55 // 100), iter_fw_edu_part4()),
    ]
    id_sources = [
        ("fineweb2_hq_id", id_total, iter_fw_id_part4()),
    ]

    t0 = time.time()
    pool = (
        mp.Pool(processes=args.workers, initializer=_worker_init, initargs=(str(args.tokenizer),))
        if args.workers > 1
        else None
    )
    try:
        print(f"[part4] Packing {en_total:,} EN + {id_total:,} ID train tokens -> {args.out}", flush=True)
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
        "corpus": "kuon2-pretrain-1b-part4",
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
        f"[part4] DONE in {meta['elapsed_sec']}s | "
        f"EN {en_stats['train']:,} train / {en_stats['val']:,} val | "
        f"ID {id_stats['train']:,} train / {id_stats['val']:,} val",
        flush=True,
    )


if __name__ == "__main__":
    main()

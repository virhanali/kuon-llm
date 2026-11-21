"""Prepare a Kuon dataset: (optionally) train the tokenizer, then pack
every document into train/val uint16 token streams.

Usage:
    python scripts/prepare_data.py --source tinystories --out data/tinystories
    python scripts/prepare_data.py --source file --path corpus.txt --out data/mycorpus

Outputs under --out:
    tokenizer.json   the exact tokenizer used (reused on later runs)
    train.bin        packed token stream (documents joined by <endoftext>)
    val.bin          first --val-docs documents, held out for evaluation
    meta.json        vocab size + corpus stats

Encoding is parallelized across CPU cores; documents are streamed, so
RAM usage stays flat regardless of corpus size.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from tqdm import tqdm

from kuon.data import BinWriter
from kuon.tokenizer import BPETokenizer

_TOK: BPETokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _TOK
    _TOK = BPETokenizer.load(tok_path)


def _encode_doc(text: str) -> list[int]:
    assert _TOK is not None
    return _TOK.encode(text, add_eos=True)


def iter_tinystories(limit: int | None):
    """TinyStories from the Hugging Face hub (~2M simple English stories)."""
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train")
    total = min(len(ds), limit) if limit else len(ds)
    for i, row in enumerate(ds):
        if i >= total:
            break
        for key in ("story", "text", "content"):
            if isinstance(row.get(key), str):
                yield row[key]
                break
        else:
            raise SystemExit(f"unexpected TinyStories columns: {list(row.keys())}")


def iter_file_paragraphs(path: str, limit: int | None):
    """A local text file; paragraphs separated by blank lines are documents."""
    text = Path(path).read_text(encoding="utf-8")
    docs = [p for p in text.split("\n\n") if p.strip()]
    if limit:
        docs = docs[:limit]
    yield from docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["tinystories", "file"], default="tinystories")
    ap.add_argument("--path", help="text file for --source file")
    ap.add_argument("--out", default="data/tinystories")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--val-docs", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on total documents (quick runs)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tokenizer-sample-chars", type=int, default=100_000_000,
                    help="chars of the corpus used to train the tokenizer")
    ap.add_argument("--force-tokenizer", action="store_true",
                    help="retrain the tokenizer even if tokenizer.json exists")
    args = ap.parse_args()

    if args.source == "file" and not args.path:
        raise SystemExit("--source file requires --path")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def train_docs():
        if args.source == "tinystories":
            return iter_tinystories(args.limit)
        return iter_file_paragraphs(args.path, args.limit)

    # 1) tokenizer: reuse if present so datasets stay compatible across runs
    tok_path = out / "tokenizer.json"
    if tok_path.exists() and not args.force_tokenizer:
        tok = BPETokenizer.load(tok_path)
        print(f"[data] reusing tokenizer at {tok_path} (vocab_size={tok.vocab_size})")
    else:
        print(f"[data] training tokenizer (vocab_size={args.vocab_size}) ...")
        tok = BPETokenizer.train(train_docs(), vocab_size=args.vocab_size,
                                 sample_limit_chars=args.tokenizer_sample_chars)
        tok.save(tok_path)
        print(f"[data] tokenizer saved (vocab_size={tok.vocab_size})")

    # 2) pack documents into train/val bins, val = first val_docs documents
    train_w = BinWriter(out / "train.bin")
    val_w = BinWriter(out / "val.bin")
    n_docs = 0

    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(str(tok_path),))
    try:
        for i, ids in enumerate(pool.imap(_encode_doc, train_docs(), chunksize=128)):
            (val_w if i < args.val_docs else train_w).write(ids)
            n_docs += 1
            if n_docs % 20000 == 0:
                print(f"[data] {n_docs:,} docs, {train_w.n_tokens:,} train tokens")
    finally:
        pool.close()
        pool.join()
    train_w.close()
    val_w.close()

    (out / "meta.json").write_text(json.dumps({
        "source": args.source,
        "vocab_size": tok.vocab_size,
        "n_docs": n_docs,
        "train_tokens": train_w.n_tokens,
        "val_tokens": val_w.n_tokens,
    }))
    print(f"[data] done: {n_docs:,} docs | train {train_w.n_tokens:,} tokens | "
          f"val {val_w.n_tokens:,} tokens | -> {out}/")


if __name__ == "__main__":
    main()

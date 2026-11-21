"""Stream-pack a large text corpus into uint16 token bins.

Usage:
    python scripts/pack_corpus.py --input /kaggle/input/.../cosmopedia.txt \
        --tokenizer tokenizers/cosmopedia-16k.json --out data/cosmopedia \
        --max-tokens 400000000 --workers 2
"""

from __future__ import annotations

import argparse

from kuon.data import pack_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="plain text file, blank-line separated docs")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="data/cosmopedia")
    ap.add_argument("--val-docs", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    pack_corpus(args.input, args.tokenizer, args.out, val_docs=args.val_docs,
                workers=args.workers, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()

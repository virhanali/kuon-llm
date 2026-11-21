"""Create a small synthetic instruction dataset from a TinyStories corpus.

Usage:
    python scripts/prepare_sft.py --corpus /kaggle/working/corpus.txt --out data/sft --n 3000

Pairs are simple prompts mapped to real stories from the corpus, e.g.
    {"user": "Tell me a short story about a cat.", "assistant": "Once upon a time ..."}
This is intentionally minimal: it teaches the chat *format* (prompt ->
masked answer). Real instruction data can be dropped into the same JSONL
schema later.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROMPTS = [
    ("Tell me a story.", None),
    ("Tell me a short story.", None),
    ("Write a story about a {theme}.", None),
    ("Can you tell me a story about a {theme}?", None),
]
THEMES = ["cat", "dog", "girl", "boy", "dragon", "robot", "friend", "family",
          "park", "beach", "bird", "bear", "bunny", "train", "garden"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="plain text file, stories separated by blank lines")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--n", type=int, default=3000, help="number of pairs")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = [d.strip() for d in Path(args.corpus).read_text().split("\n\n") if d.strip()]
    if not docs:
        raise SystemExit("corpus is empty")

    # index stories by theme so themed prompts get thematically matching stories
    by_theme = {}
    for theme in THEMES:
        needle = f" {theme}"
        matches = [d for d in docs if needle in d.lower()]
        if matches:
            by_theme[theme] = matches

    pairs = []
    for _ in range(args.n):
        template, _ = rng.choice(PROMPTS)
        theme = rng.choice(list(by_theme))
        story = rng.choice(by_theme[theme])
        pairs.append({"user": template.format(theme=theme), "assistant": story})

    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("sft_train.jsonl", pairs[n_val:]), ("sft_val.jsonl", pairs[:n_val])):
        with (out / name).open("w") as f:
            for pair in chunk:
                f.write(json.dumps(pair) + "\n")
        print(f"[sft-data] {name}: {len(chunk)} pairs")

    print(f"[sft-data] done -> {out}/  (themes available: {len(by_theme)})")


if __name__ == "__main__":
    main()

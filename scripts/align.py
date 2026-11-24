"""Preference alignment CLI (APO-zero / ORPO) for Kuon models.

Usage:
    python scripts/align.py --base checkpoints/kuon-1-sft.pt \
        --tokenizer tokenizers/kuon1-bilingual-24k.json \
        --data data/kuon1-prefs --out checkpoints/kuon-1-aligned.pt \
        --method apo_zero --steps 300 --lr 1e-5 --beta 2.0
"""

from __future__ import annotations

import argparse

from kuon.align import align


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="SFT or pretrained checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="dir with prefs_train.jsonl / prefs_val.jsonl")
    ap.add_argument("--out", required=True, help="output aligned checkpoint path")
    ap.add_argument("--tokenizer", help="exact tokenizer used by the checkpoint")
    ap.add_argument("--method", choices=("apo_zero", "orpo"), default="apo_zero")
    ap.add_argument("--business", action="store_true", help="default business prompt flag")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--sft-weight", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    align(
        args.base, args.data, out_path=args.out, method=args.method,
        steps=args.steps, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, beta=args.beta, sft_weight=args.sft_weight,
        eval_interval=args.eval_interval, device=args.device, seed=args.seed,
        tokenizer_path=args.tokenizer, business=args.business, quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

"""Supervised fine-tuning (SFT) of a pretrained Kuon checkpoint.

Usage:
    python scripts/sft.py --base checkpoints/kuon-13m/kuon-13m-last.pt \
        --data data/sft --out checkpoints/kuon-13m-sft.pt --steps 300
"""

from __future__ import annotations

import argparse

from kuon.sft import sft


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="pretrained checkpoint (.pt)")
    ap.add_argument("--data", default="data/sft", help="dir with sft_train.jsonl / sft_val.jsonl")
    ap.add_argument("--out", default=None, help="output checkpoint path")
    ap.add_argument("--tokenizer", help="exact tokenizer used by the base checkpoint")
    ap.add_argument("--business", action="store_true", help="use the business chat prompt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-interval", type=int, default=0,
                    help="evaluate val_loss every N steps and save best checkpoint (0=final only)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after N consecutive evals without val_loss improvement (0=off)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-prompt", default="Tell me a story.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sft(args.base, args.data, out_path=args.out, steps=args.steps,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, eval_interval=args.eval_interval,
        patience=args.patience, device=args.device,
        seed=args.seed, sample_prompt=args.sample_prompt, quiet=args.quiet,
        tokenizer_path=args.tokenizer, business=args.business)


if __name__ == "__main__":
    main()

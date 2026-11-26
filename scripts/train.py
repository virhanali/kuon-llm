"""Train a Kuon model from a config file.

Usage:
    python scripts/train.py --config configs/kuon-13m.json
    python scripts/train.py --config configs/kuon-1-loop.json --device cuda --steps 5000
    python scripts/train.py --config configs/kuon-13m.json --resume checkpoints/kuon-13m/kuon-13m-last.pt
    torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/kuon-1-loop.json \
        --upcycle-from checkpoints/kuon-1-dense/kuon-1-dense-best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from kuon.config import load_run_config
from kuon.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/kuon-13m.json")
    ap.add_argument("--data-dir", default=None, help="override the dataset directory")
    ap.add_argument("--steps", type=int, default=None, help="override total steps (defines cosine LR horizon)")
    ap.add_argument("--stop-step", type=int, default=None,
                    help="optional early stop step (saves checkpoint cleanly while keeping cosine schedule over --steps)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="optional wall-clock budget in seconds (saves checkpoint cleanly on time-box expiry)")
    ap.add_argument("--batch-size", type=int, default=None, help="override microbatch size per GPU")
    ap.add_argument("--grad-accum", type=int, default=None, help="override gradient accumulation steps")
    ap.add_argument("--eval-interval", type=int, default=None, help="override evaluation interval")
    ap.add_argument("--lr", type=float, default=None, help="override peak learning rate")
    ap.add_argument("--min-lr", type=float, default=None, help="override minimum learning rate")
    ap.add_argument("--warmup-steps", type=int, default=None, help="override warmup steps")
    ap.add_argument("--checkpoint-dir", default=None, help="override checkpoint output directory")
    ap.add_argument("--log-file", default=None, help="override JSONL log output file")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (restores optimizer & step)")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint weights to initialize continual pretraining with a fresh LR schedule")
    ap.add_argument("--upcycle-from", default=None,
                    help="dense checkpoint to zero-loss upcycle into a gated-looped model")
    ap.add_argument("--schedule-start", type=int, default=0,
                    help="step offset where the cosine LR schedule starts")
    ap.add_argument("--compile", action="store_true", help="torch.compile (cuda)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="enable activation checkpointing to save VRAM")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="disable activation checkpointing for speed on 24GB+ GPUs")
    ap.add_argument("--quiet", action="store_true", help="no progress samples")
    args = ap.parse_args()

    name, mcfg, tcfg = load_run_config(args.config)
    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["grad_accum"] = args.grad_accum
    if args.eval_interval is not None:
        overrides["eval_interval"] = args.eval_interval
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.min_lr is not None:
        overrides["min_lr"] = args.min_lr
    if args.warmup_steps is not None:
        overrides["warmup_steps"] = args.warmup_steps
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.log_file is not None:
        overrides["log_file"] = args.log_file
    if args.grad_checkpoint and args.no_grad_checkpoint:
        ap.error("cannot set both --grad-checkpoint and --no-grad-checkpoint")
    if args.grad_checkpoint:
        overrides["grad_checkpoint"] = True
    elif args.no_grad_checkpoint:
        overrides["grad_checkpoint"] = False
    if overrides:
        tcfg = replace(tcfg, **overrides)

    train(name, mcfg, tcfg, data_dir=args.data_dir, steps=args.steps,
          stop_step=args.stop_step, max_seconds=args.max_seconds,
          device=args.device, resume=args.resume, init_from=args.init_from,
          upcycle_from=args.upcycle_from, schedule_start=args.schedule_start,
          compile_=args.compile or tcfg.compile, quiet=args.quiet)


if __name__ == "__main__":
    main()

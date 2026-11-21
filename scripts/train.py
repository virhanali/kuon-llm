"""Train a Kuon model from a config file.

Usage:
    python scripts/train.py --config configs/kuon-13m.json
    python scripts/train.py --config configs/kuon-48m.json --device cuda --steps 5000
    python scripts/train.py --config configs/kuon-13m.json --resume checkpoints/kuon-13m/kuon-13m-last.pt
"""

from __future__ import annotations

import argparse

from kuon.config import load_run_config
from kuon.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/kuon-13m.json")
    ap.add_argument("--data-dir", default=None, help="override the dataset directory")
    ap.add_argument("--steps", type=int, default=None, help="override total steps")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--compile", action="store_true", help="torch.compile (cuda)")
    ap.add_argument("--quiet", action="store_true", help="no progress samples")
    args = ap.parse_args()

    name, mcfg, tcfg = load_run_config(args.config)
    train(name, mcfg, tcfg, data_dir=args.data_dir, steps=args.steps,
          device=args.device, resume=args.resume,
          compile_=args.compile or tcfg.compile, quiet=args.quiet)


if __name__ == "__main__":
    main()

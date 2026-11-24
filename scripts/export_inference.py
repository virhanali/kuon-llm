"""Export a lossless inference-only Kuon checkpoint by removing optimizer state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="original training checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="new inference-only checkpoint")
    args = parser.parse_args()

    if args.checkpoint.resolve() == args.out.resolve():
        parser.error("--out must differ from --checkpoint")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "model" not in state or "model_cfg" not in state:
        raise ValueError("not a Kuon checkpoint: expected model and model_cfg")
    state.pop("optimizer", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = args.out.open("xb")
    try:
        with output:
            torch.save(state, output)
    except BaseException:
        args.out.unlink(missing_ok=True)
        raise

    before = args.checkpoint.stat().st_size
    after = args.out.stat().st_size
    print(f"Training checkpoint:  {before:,} bytes")
    print(f"Inference checkpoint: {after:,} bytes")
    print(f"Saved: {before - after:,} bytes ({100 * (before - after) / before:.1f}%)")
    print("Model weights are unchanged (lossless).")
    print("Keep the original to resume optimizer training.")


if __name__ == "__main__":
    main()

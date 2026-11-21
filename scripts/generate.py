"""Generate text from a trained Kuon checkpoint (streams as it decodes).

Usage:
    python scripts/generate.py --checkpoint checkpoints/kuon-13m/kuon-13m-best.pt \
        --prompt "Once upon a time" --tokens 200
    python scripts/generate.py --checkpoint ... --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kuon.config import ModelConfig
from kuon.generate import generate
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer, StreamDecoder


def load_model(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**state["model_cfg"])
    model = Transformer(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    return model, state


def find_tokenizer(state, args) -> BPETokenizer:
    if args.tokenizer:
        return BPETokenizer.load(args.tokenizer)
    data_dir = state.get("train_cfg", {}).get("data_dir", "data/tinystories")
    path = Path(data_dir) / "tokenizer.json"
    if path.exists():
        return BPETokenizer.load(path)
    raise SystemExit(f"tokenizer not found at {path}; pass --tokenizer")


def run_once(model, tok, args, device) -> None:
    ids = tok.encode(args.prompt)
    if not ids:
        ids = [tok.eos_id] if tok.eos_id is not None else [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    dec = StreamDecoder(tok)
    print(f"> {args.prompt}", flush=True)
    for new in generate(model, x, max_new_tokens=args.tokens,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, eos_id=tok.eos_id):
        if tok.eos_id in new:
            break
        print(dec.push(new[0]), end="", flush=True)
    print("\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--tokenizer", default=None, help="override tokenizer path")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    tok = find_tokenizer(state, args)

    if args.interactive:
        print("Kuon interactive session. Empty line or Ctrl-D exits.")
        while True:
            try:
                prompt = input("you> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            args.prompt = prompt
            run_once(model, tok, args, device)
    else:
        run_once(model, tok, args, device)


if __name__ == "__main__":
    main()

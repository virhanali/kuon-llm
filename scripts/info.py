"""Print the parameter / memory profile of a Kuon config.

Usage:
    python scripts/info.py --config configs/kuon-13m.json
"""

from __future__ import annotations

import argparse

from kuon.config import load_run_config
from kuon.model import Transformer
from kuon.utils import param_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/kuon-13m.json")
    args = ap.parse_args()

    name, mcfg, _ = load_run_config(args.config)
    model = Transformer(mcfg)
    n = param_count(model)

    C, hd = mcfg.n_embd, mcfg.head_dim
    attn = 2 * C * C + 2 * mcfg.n_kv_head * hd * C
    ffn = 3 * C * mcfg.ffn_hidden
    per_layer = attn + ffn
    embed = mcfg.vocab_size * C

    kv_bytes_per_token = 2 * mcfg.n_layer * mcfg.n_kv_head * hd * 4  # fp32 k+v

    print(f"config            {name}")
    print(f"layers             {mcfg.n_layer}")
    print(f"n_embd             {C}")
    print(f"heads (q / kv)     {mcfg.n_head} / {mcfg.n_kv_head}  (head_dim={hd}, GQA x{mcfg.n_rep})")
    print(f"ffn_hidden         {mcfg.ffn_hidden}")
    print(f"context            {mcfg.block_size} tokens")
    print(f"embedding          {embed/1e6:.2f}M  (tied lm_head: {mcfg.tie_word_embeddings})")
    print(f"per layer          attn {attn/1e6:.2f}M + ffn {ffn/1e6:.2f}M = {per_layer/1e6:.2f}M")
    print(f"total params       {n/1e6:.2f}M")
    print(f"fp32 weights       {n*4/1e6:.1f} MB  (AdamW adds ~2x during training)")
    print(f"kv cache           {kv_bytes_per_token/1024:.1f} KB/token "
          f"-> {kv_bytes_per_token*mcfg.block_size/1e6:.0f} MB at full context")


if __name__ == "__main__":
    main()

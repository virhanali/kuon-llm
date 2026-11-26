"""Export a Kuon checkpoint to portable per-channel INT8 (~3.7x smaller, ~2.2x faster CPU decode).

Usage:
    PYTHONPATH=. python scripts/quantize_cpu.py \
        --checkpoint checkpoints/kuon-2-inference.pt \
        --out checkpoints/kuon-2-int8.pt \
        --threads 4 --bench
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from kuon.config import ModelConfig
from kuon.generate import generate
from kuon.model import Transformer
from kuon.quant import (
    QUANT_FORMAT_INT8,
    apply_cpu_int8_runtime,
    dequantize_state_dict_int8,
    quantize_state_dict_int8,
)


def export_int8_checkpoint(src: Path, dst: Path) -> dict:
    if src.resolve() == dst.resolve():
        raise ValueError("--out must differ from --checkpoint")
    state = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state or "model_cfg" not in state:
        raise ValueError("not a valid Kuon checkpoint (expected model and model_cfg)")
    if state.get("quant") == QUANT_FORMAT_INT8:
        raise ValueError("checkpoint is already quantized")

    q_payload, stats = quantize_state_dict_int8(state["model"])
    out_state = {
        "quant": QUANT_FORMAT_INT8,
        "model_quant": q_payload,
        "model_cfg": state["model_cfg"],
        "train_cfg": state.get("train_cfg", {}),
        "step": state.get("step", 0),
        "val_loss": state.get("val_loss"),
    }
    for meta_key in ("sft_meta", "align_meta"):
        if meta_key in state:
            out_state[meta_key] = state[meta_key]

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".tmp")
    torch.save(out_state, tmp)
    os.replace(tmp, dst)

    tok_src = src.parent / "tokenizer.json"
    tok_dst = dst.parent / "tokenizer.json"
    if tok_src.is_file() and tok_src.resolve() != tok_dst.resolve():
        shutil.copyfile(tok_src, tok_dst)

    return {
        **stats,
        "src_file_bytes": src.stat().st_size,
        "dst_file_bytes": dst.stat().st_size,
    }


@torch.inference_mode()
def verify_and_bench(
    src: Path,
    dst: Path,
    *,
    threads: int = 4,
    decode_tokens: int = 24,
    dyn_int8: bool = False,
) -> dict:
    torch.set_num_threads(threads)
    fp32_state = torch.load(src, map_location="cpu", weights_only=False)
    q_state = torch.load(dst, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**fp32_state["model_cfg"])

    m_fp32 = Transformer(cfg).eval()
    m_fp32.load_state_dict(fp32_state["model"])

    m_runtime = Transformer(cfg).eval()
    m_runtime.load_state_dict(
        dequantize_state_dict_int8(q_state["model_quant"], tie_word_embeddings=cfg.tie_word_embeddings)
    )
    if dyn_int8:
        m_runtime = apply_cpu_int8_runtime(m_runtime)

    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (1, min(64, cfg.block_size)))
    y_fp32, _ = m_fp32(x)
    y_run, _ = m_runtime(x)
    mean_diff = (y_fp32 - y_run).abs().mean().item()
    top1_match = (y_fp32.argmax(dim=-1) == y_run.argmax(dim=-1)).float().mean().item()

    list(generate(m_runtime, x[:, :16], max_new_tokens=2, temperature=0.0))
    t0 = time.time()
    list(generate(m_runtime, x, max_new_tokens=decode_tokens, temperature=0.0))
    dt_run = time.time() - t0

    return {
        "mean_abs_logit_diff": round(mean_diff, 5),
        "top1_agreement_pct": round(100.0 * top1_match, 2),
        "int8_decode_tok_s": round(decode_tokens / max(dt_run, 1e-6), 2),
        "threads": threads,
        "dyn_int8": dyn_int8,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="FP32 Kuon checkpoint (.pt)")
    ap.add_argument("--out", type=Path, required=True, help="output INT8 checkpoint (.pt)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for optional benchmark")
    ap.add_argument("--bench", action="store_true", help="verify logit fidelity and measure CPU decode tok/s")
    ap.add_argument("--dyn-int8", action="store_true", help="also apply selective dynamic INT8 Linear kernels during benchmark")
    args = ap.parse_args()

    stats = export_int8_checkpoint(args.checkpoint, args.out)
    ratio = stats["src_file_bytes"] / max(stats["dst_file_bytes"], 1)
    print(
        f"[quantize-cpu] Exported {args.out} | "
        f"{stats['src_file_bytes']/1e6:.1f} MB -> {stats['dst_file_bytes']/1e6:.1f} MB ({ratio:.2f}x smaller) "
        f"| quantized_tensors={stats['quantized_tensors']}",
        flush=True,
    )
    if args.bench:
        res = verify_and_bench(args.checkpoint, args.out, threads=args.threads, dyn_int8=args.dyn_int8)
        print(
            f"[quantize-cpu] Fidelity: mean_abs_logit_diff={res['mean_abs_logit_diff']} "
            f"| top1_agreement={res['top1_agreement_pct']}% "
            f"| CPU decode ({res['threads']} threads)={res['int8_decode_tok_s']} tok/s",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Preference alignment (APO-zero and ORPO) for Kuon models.

Implements two small-model-friendly alignment objectives grounded in 2024-2026 research:
  1. `apo_zero` — Anchored Preference Optimization (D'Oosterlinck et al., TACL 2025;
     used by Hugging Face SmolLM3). Explicitly anchors chosen response log-likelihood
     upward and rejected response log-likelihood downward relative to a frozen
     reference model, preventing the probability collapse common in standard DPO on
     ~120M parameter models. Reference log-probs are precomputed once before training,
     so GPU memory overhead during optimization is zero.
  2. `orpo` — Odds Ratio Preference Optimization (Hong et al., EMNLP 2024). Reference-free
     single-stage SFT NLL + log-odds margin between chosen and rejected responses.

Both use length-normalized average log-probabilities over assistant tokens only,
preventing length bias toward short canned refusals ("I do not know.").
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from kuon.chat import build_example
from kuon.config import ModelConfig
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer
from kuon.utils import seed_all


def load_pref_pairs(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _encode_pref(tokenizer: BPETokenizer, row: dict, block: int, default_business: bool):
    biz = bool(row.get("business", default_business))
    ev = row.get("evidence")
    wx, wy = build_example(tokenizer, row["user"], row["chosen"], block, evidence=ev, business=biz)
    lx, ly = build_example(tokenizer, row["user"], row["rejected"], block, evidence=ev, business=biz)
    if not wx or not lx or not any(v != -100 for v in wy) or not any(v != -100 for v in ly):
        return None
    return (wx, wy, lx, ly)


def _pad_seqs(seqs: list[tuple[list[int], list[int]]], pad_id: int, device: str):
    max_len = max(len(x) for x, _ in seqs)
    x = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids, _ in seqs], dtype=torch.long, device=device)
    y = torch.tensor([lbl + [-100] * (max_len - len(lbl)) for _, lbl in seqs], dtype=torch.long, device=device)
    return x, y


def _avg_logprobs(model: Transformer, x: torch.Tensor, y: torch.Tensor,
                  amp_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Length-normalized average log-probability per sequence over assistant tokens."""
    if amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(x)
    else:
        logits, _ = model(x)
    # Compute log-softmax in FP32 for numerical stability in preference ratios
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = (y != -100)
    safe_y = y.masked_fill(~mask, 0)
    tok_logp = logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask.float()
    counts = mask.float().sum(dim=-1).clamp_min(1.0)
    return tok_logp.sum(dim=-1) / counts


@torch.no_grad()
def _precompute_ref_logps(
    model: Transformer,
    encoded: list[tuple[list[int], list[int], list[int], list[int]]],
    batch_size: int,
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype | None,
) -> list[tuple[float, float]]:
    model.eval()
    out: list[tuple[float, float]] = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        wx, wy = _pad_seqs([(c[0], c[1]) for c in chunk], pad_id, device)
        lx, ly = _pad_seqs([(c[2], c[3]) for c in chunk], pad_id, device)
        rw = _avg_logprobs(model, wx, wy, amp_dtype=amp_dtype).cpu().tolist()
        rl = _avg_logprobs(model, lx, ly, amp_dtype=amp_dtype).cpu().tolist()
        out.extend(zip(rw, rl))
    model.train()
    return out


@torch.no_grad()
def _eval_prefs(
    model: Transformer,
    encoded: list[tuple[list[int], list[int], list[int], list[int]]],
    ref_logps: list[tuple[float, float]] | None,
    *,
    method: str,
    beta: float,
    sft_weight: float,
    batch_size: int,
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    n = 0
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        wx, wy = _pad_seqs([(c[0], c[1]) for c in chunk], pad_id, device)
        lx, ly = _pad_seqs([(c[2], c[3]) for c in chunk], pad_id, device)
        pw = _avg_logprobs(model, wx, wy, amp_dtype=amp_dtype)
        pl = _avg_logprobs(model, lx, ly, amp_dtype=amp_dtype)
        margin = pw - pl
        if method == "apo_zero":
            assert ref_logps is not None
            rchunk = ref_logps[start:start + batch_size]
            rw = torch.tensor([r[0] for r in rchunk], dtype=torch.float32, device=device)
            rl = torch.tensor([r[1] for r in rchunk], dtype=torch.float32, device=device)
            dw = beta * (pw - rw)
            dl = beta * (pl - rl)
            # APO-zero: push chosen probability up and rejected probability down relative to ref
            loss = (1.0 - torch.sigmoid(dw) + torch.sigmoid(dl)).mean()
            if sft_weight > 0:
                loss = loss + sft_weight * (-pw).mean()
        else:
            log_odds_w = pw - torch.log1p(-torch.exp(pw.clamp(max=-1e-6)))
            log_odds_l = pl - torch.log1p(-torch.exp(pl.clamp(max=-1e-6)))
            loss = (-pw).mean() + beta * (-F.logsigmoid(log_odds_w - log_odds_l)).mean()
        bsz = len(chunk)
        total_loss += loss.item() * bsz
        total_acc += (margin > 0).float().sum().item()
        total_margin += margin.sum().item()
        n += bsz
    model.train()
    return {
        "val_loss": total_loss / max(n, 1),
        "val_pref_acc": total_acc / max(n, 1),
        "val_margin": total_margin / max(n, 1),
    }


def align(
    base_ckpt: str,
    data_dir: str,
    *,
    out_path: str | None = None,
    method: str = "apo_zero",
    steps: int = 300,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-5,
    min_lr: float = 1e-6,
    warmup_steps: int = 20,
    weight_decay: float = 0.01,
    beta: float = 2.0,
    sft_weight: float = 0.1,
    eval_interval: int = 50,
    block_size: int | None = None,
    device: str | None = None,
    seed: int = 0,
    tokenizer_path: str | Path | None = None,
    business: bool = False,
    quiet: bool = False,
) -> dict:
    if method not in ("apo_zero", "orpo"):
        raise ValueError(f"unsupported alignment method: {method}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)

    state = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**state["model_cfg"])
    if tokenizer_path is None:
        tokenizer_path = Path(base_ckpt).parent / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = Path(state["train_cfg"]["data_dir"]) / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match checkpoint")
    block = block_size or model_cfg.block_size
    pad_id = tokenizer.eos_id if tokenizer.eos_id is not None else 0

    data = Path(data_dir)
    train_enc = [e for r in load_pref_pairs(data / "prefs_train.jsonl")
                 if (e := _encode_pref(tokenizer, r, block, business)) is not None]
    val_path = data / "prefs_val.jsonl"
    val_enc = [e for r in load_pref_pairs(val_path)
               if (e := _encode_pref(tokenizer, r, block, business)) is not None] if val_path.is_file() else []
    assert train_enc, "no usable preference pairs in prefs_train.jsonl"

    is_cuda = device.startswith("cuda")
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(state["model"])

    hw_bf16 = False
    if is_cuda and torch.cuda.get_device_capability()[0] >= 8:
        try:
            hw_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            hw_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = (torch.bfloat16 if hw_bf16 else torch.float16) if is_cuda else None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    train_ref = None
    val_ref = None
    if method == "apo_zero":
        t0 = time.time()
        train_ref = _precompute_ref_logps(model, train_enc, batch_size * 2, pad_id, device, amp_dtype)
        if val_enc:
            val_ref = _precompute_ref_logps(model, val_enc, batch_size * 2, pad_id, device, amp_dtype)
        if not quiet:
            print(f"[align] precomputed {len(train_ref)} train / {len(val_ref or [])} val ref logps "
                  f"in {time.time() - t0:.1f}s", flush=True)

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=is_cuda,
    )

    baseline_val = _eval_prefs(
        model, val_enc, val_ref, method=method, beta=beta, sft_weight=sft_weight,
        batch_size=batch_size * 2, pad_id=pad_id, device=device, amp_dtype=amp_dtype,
    ) if val_enc else {"val_loss": float("nan"), "val_pref_acc": float("nan"), "val_margin": float("nan")}
    if not quiet:
        print(f"[align] method={method} | {len(train_enc)} train, {len(val_enc)} val pairs "
              f"| steps={steps} bs={batch_size}x{grad_accum} lr={lr:.1e} beta={beta}", flush=True)
        if val_enc:
            print(f"[align] baseline step 0 | val_loss {baseline_val['val_loss']:.4f} | "
                  f"val_pref_acc {baseline_val['val_pref_acc']:.3f} | "
                  f"val_margin {baseline_val['val_margin']:+.4f}", flush=True)

    best_score = (baseline_val["val_pref_acc"], -baseline_val["val_loss"]) if val_enc else None
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if val_enc else None
    best_metrics = dict(baseline_val)
    best_step = 0

    rng = random.Random(seed)
    order: list[int] = []
    pos = 0

    def next_indices(k: int) -> list[int]:
        nonlocal order, pos
        out_idx: list[int] = []
        while len(out_idx) < k:
            if pos >= len(order):
                order = list(range(len(train_enc)))
                rng.shuffle(order)
                pos = 0
            out_idx.append(order[pos])
            pos += 1
        return out_idx

    first_loss = None
    running_loss = None
    running_acc = None
    start_time = time.time()
    model.train()

    for step in range(steps):
        if warmup_steps > 0 and step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_acc = 0.0

        for _ in range(grad_accum):
            idxs = next_indices(batch_size)
            chunk = [train_enc[i] for i in idxs]
            wx, wy = _pad_seqs([(c[0], c[1]) for c in chunk], pad_id, device)
            lx, ly = _pad_seqs([(c[2], c[3]) for c in chunk], pad_id, device)
            pw = _avg_logprobs(model, wx, wy, amp_dtype=amp_dtype)
            pl = _avg_logprobs(model, lx, ly, amp_dtype=amp_dtype)

            if method == "apo_zero":
                assert train_ref is not None
                rw = torch.tensor([train_ref[i][0] for i in idxs], dtype=torch.float32, device=device)
                rl = torch.tensor([train_ref[i][1] for i in idxs], dtype=torch.float32, device=device)
                dw = beta * (pw - rw)
                dl = beta * (pl - rl)
                loss = (1.0 - torch.sigmoid(dw) + torch.sigmoid(dl)).mean()
                if sft_weight > 0:
                    loss = loss + sft_weight * (-pw).mean()
            else:
                log_odds_w = pw - torch.log1p(-torch.exp(pw.clamp(max=-1e-6)))
                log_odds_l = pl - torch.log1p(-torch.exp(pl.clamp(max=-1e-6)))
                loss = (-pw).mean() + beta * (-F.logsigmoid(log_odds_w - log_odds_l)).mean()

            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            step_loss += loss.item() / grad_accum
            step_acc += (pw.detach() > pl.detach()).float().mean().item() / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        first_loss = step_loss if first_loss is None else first_loss
        running_loss = step_loss if running_loss is None else 0.9 * running_loss + 0.1 * step_loss
        running_acc = step_acc if running_acc is None else 0.9 * running_acc + 0.1 * step_acc

        do_eval = (eval_interval > 0 and (step + 1) % eval_interval == 0) or (step + 1 == steps)
        if do_eval:
            if val_enc:
                metrics = _eval_prefs(
                    model, val_enc, val_ref, method=method, beta=beta, sft_weight=sft_weight,
                    batch_size=batch_size * 2, pad_id=pad_id, device=device, amp_dtype=amp_dtype,
                )
                score = (metrics["val_pref_acc"], -metrics["val_loss"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = dict(metrics)
                    best_step = step + 1
                if not quiet:
                    print(f"step {step+1:4d}/{steps} | loss {running_loss:.4f} | train_acc {running_acc:.3f} | "
                          f"val_loss {metrics['val_loss']:.4f} | val_acc {metrics['val_pref_acc']:.3f} | "
                          f"val_margin {metrics['val_margin']:+.4f} | lr {cur_lr:.2e} | "
                          f"{time.time() - start_time:.0f}s", flush=True)
            elif not quiet:
                print(f"step {step+1:4d}/{steps} | loss {running_loss:.4f} | train_acc {running_acc:.3f} | "
                      f"lr {cur_lr:.2e} | {time.time() - start_time:.0f}s", flush=True)

    final_state = best_state if best_state is not None else model.state_dict()
    out = Path(out_path or f"checkpoints/kuon-{method}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    torch.save({
        "model": final_state,
        "model_cfg": asdict(model_cfg),
        "train_cfg": state.get("train_cfg", {}),
        "step": state.get("step", 0),
        "val_loss": best_metrics.get("val_loss", float("nan")),
        "align_meta": {
            "method": method,
            "best_step": best_step,
            "val_pref_acc": best_metrics.get("val_pref_acc"),
            "val_margin": best_metrics.get("val_margin"),
        },
    }, tmp)
    os.replace(tmp, out)
    dest_tok = out.parent / "tokenizer.json"
    if Path(tokenizer_path).resolve() != dest_tok.resolve():
        shutil.copyfile(tokenizer_path, dest_tok)

    if not quiet:
        print(f"[align] done | best_step={best_step} | val_acc={best_metrics.get('val_pref_acc', float('nan')):.3f} "
              f"| val_margin={best_metrics.get('val_margin', float('nan')):+.4f} | saved {out}", flush=True)

    return {
        "first_loss": first_loss,
        "last_loss": running_loss,
        "baseline_val_acc": baseline_val["val_pref_acc"],
        "best_val_acc": best_metrics["val_pref_acc"],
        "best_val_margin": best_metrics["val_margin"],
        "best_val_loss": best_metrics["val_loss"],
        "best_step": best_step,
        "ckpt": str(out),
        "steps": steps,
    }

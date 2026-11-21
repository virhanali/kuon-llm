"""Kuon trainer: next-token pretraining with AdamW + cosine schedule.

The whole loop is deliberately plain: sample random crops from the
packed token stream, forward/backward with gradient accumulation, clip,
step, log, evaluate, checkpoint. Everything an LLM run needs at this
scale and nothing else.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from kuon.data import get_batch, load_meta, open_bin
from kuon.generate import sample_text
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer
from kuon.utils import JsonlLogger, decay_param_groups, param_count, seed_all


def get_lr(step: int, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def estimate_val_loss(model, val_data, block_size: int, batch_size: int,
                      device: str, batches: int) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(val_data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: str | Path, model, model_cfg, train_cfg, step: int,
                    optimizer, val_loss: float) -> None:
    """Atomic checkpoint write (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, tmp)
    os.replace(tmp, path)


def train(name: str, model_cfg, train_cfg, *, data_dir: str | None = None,
          steps: int | None = None, device: str | None = None,
          resume: str | None = None, compile_: bool = False, quiet: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    steps = steps or train_cfg.steps
    data_dir = data_dir or train_cfg.data_dir
    train_cfg = replace(train_cfg, data_dir=data_dir)  # checkpoints must record it
    seed_all(train_cfg.seed)

    # the tokenizer in the data dir defines the vocab; the model must match it
    meta = load_meta(data_dir)
    model_cfg = replace(model_cfg, vocab_size=meta["vocab_size"])
    tokenizer = BPETokenizer.load(Path(data_dir) / "tokenizer.json")
    train_data = open_bin(data_dir, "train")
    val_data = open_bin(data_dir, "val")

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    log = JsonlLogger(train_cfg.log_file)

    model = Transformer(model_cfg).to(device)
    raw_model = model
    if compile_ and device == "cuda":
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        decay_param_groups(raw_model, train_cfg.weight_decay, train_cfg.lr, train_cfg.betas),
        lr=train_cfg.lr, betas=tuple(train_cfg.betas), fused=(device == "cuda"))

    start_step, best_val = 0, float("inf")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        best_val = state.get("val_loss", float("inf"))
        if not quiet:
            print(f"[kuon] resumed {name} from {resume} (step {start_step})")

    if not quiet:
        print(f"[kuon] {name}: {param_count(model)/1e6:.2f}M params, "
              f"device={device}, steps={steps}, "
              f"tokens/step={model_cfg.block_size * train_cfg.batch_size * train_cfg.grad_accum:,}")

    block, bs, accum = model_cfg.block_size, train_cfg.batch_size, train_cfg.grad_accum
    tokens_per_step = block * bs * accum
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model.train()
    running: float | None = None
    first_loss: float | None = None
    t_start, t_iter = time.time(), time.time()

    # mixed precision: bf16 on Ampere+ GPUs, fp16 + GradScaler on older GPUs
    # (T4/P100 on free Colab/Kaggle), plain fp32 on CPU
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    for step in range(start_step, steps):
        lr = get_lr(step, train_cfg.lr, train_cfg.min_lr, train_cfg.warmup_steps, steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        dloss = 0.0
        for _ in range(accum):
            x, y = get_batch(train_data, block, bs, device)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            scaler.scale(loss / accum).backward()
            dloss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        step_loss = dloss / accum
        running = step_loss if running is None else 0.98 * running + 0.02 * step_loss
        if first_loss is None:
            first_loss = step_loss

        if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == steps:
            dt = time.time() - t_iter
            t_iter = time.time()
            interval = min(train_cfg.eval_interval, step + 1 - start_step)
            tok_s = int(tokens_per_step * interval / max(dt, 1e-9))

            val = estimate_val_loss(raw_model, val_data, block, bs, device, train_cfg.eval_batches)
            elapsed = round(time.time() - t_start, 1)
            print(f"step {step+1:6d}/{steps} | train {running:.4f} | val {val:.4f} | "
                  f"lr {lr:.2e} | {tok_s:,} tok/s | {elapsed}s", flush=True)
            log.write(name=name, step=step + 1, train_loss=round(running, 4),
                      val_loss=round(val, 4), lr=lr, tokens_per_sec=tok_s, elapsed=elapsed)

            if not quiet and train_cfg.sample_prompt and train_cfg.sample_tokens > 0:
                text = sample_text(raw_model, tokenizer, train_cfg.sample_prompt,
                                   train_cfg.sample_tokens, device=device)
                print(f"  sample> {text.strip()[:240]}", flush=True)

            save_checkpoint(ckpt_dir / f"{name}-last.pt", raw_model, model_cfg,
                            train_cfg, step + 1, opt, val)
            if val < best_val:
                best_val = val
                save_checkpoint(ckpt_dir / f"{name}-best.pt", raw_model, model_cfg,
                                train_cfg, step + 1, opt, val)

    return {
        "name": name,
        "steps": steps,
        "first_loss": first_loss,
        "last_loss": running,
        "best_val": best_val,
        "ckpt": str(ckpt_dir / f"{name}-last.pt"),
    }

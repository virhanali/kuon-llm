"""Kuon trainer: next-token pretraining with AdamW + cosine schedule.

The whole loop is deliberately plain: sample random crops from the
packed token stream, forward/backward with gradient accumulation, clip,
step, log, evaluate, checkpoint. Everything an LLM run needs at this
scale and nothing else.
"""

from __future__ import annotations

import atexit
from contextlib import nullcontext
import gc
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from kuon.data import get_batch, load_meta, open_bin
from kuon.generate import sample_text
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer
from kuon.utils import JsonlLogger, decay_param_groups, param_count, seed_all

_DDP_CLEANUP_REGISTERED = False


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


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
                      device: str, batches: int,
                      amp_dtype: torch.dtype | None = None) -> float:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(0)  # same windows per checkpoint; don't advance the training RNG
    losses = []
    for _ in range(batches):
        x, y = get_batch(val_data, block_size, batch_size, device, rng=rng)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y)
        else:
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
          steps: int | None = None, stop_step: int | None = None,
          device: str | None = None,
          resume: str | None = None, upcycle_from: str | None = None,
          init_from: str | None = None,
          compile_: bool = False, quiet: bool = False,
          schedule_start: int = 0) -> dict:
    global _DDP_CLEANUP_REGISTERED
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    is_master = rank == 0

    if ddp:
        if device is not None and not device.startswith("cuda"):
            raise ValueError("DDP training requires CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            if not _DDP_CLEANUP_REGISTERED:
                atexit.register(_cleanup_ddp)
                _DDP_CLEANUP_REGISTERED = True
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    is_cuda = device.startswith("cuda")
    steps = steps or train_cfg.steps
    data_dir = data_dir or train_cfg.data_dir
    train_cfg = replace(train_cfg, data_dir=data_dir)  # checkpoints must record it
    seed_all(train_cfg.seed)

    raw_model = None
    ddp_model = None
    model = None
    opt = None
    try:
        # the tokenizer in the data dir defines the vocab; the model must match it
        meta = load_meta(data_dir)
        model_cfg = replace(model_cfg, vocab_size=meta["vocab_size"])
        tokenizer = BPETokenizer.load(Path(data_dir) / "tokenizer.json")
        bilingual = meta.get("language_mix") == {"en": 60, "id": 40}
        cycle = ("en", "id", "en", "id", "en") if bilingual else ("",)
        if bilingual and train_cfg.grad_accum % len(cycle):
            raise ValueError("60/40 training needs grad_accum divisible by five")
        train_data = {lang: open_bin(data_dir, f"train.{lang}" if lang else "train")
                      for lang in set(cycle)}
        val_data = {lang: open_bin(data_dir, f"val.{lang}" if lang else "val")
                    for lang in set(cycle)}

        ckpt_dir = Path(train_cfg.checkpoint_dir)
        if is_master:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tok_src = Path(data_dir) / "tokenizer.json"
            tok_dst = ckpt_dir / "tokenizer.json"
            if tok_src.is_file() and tok_src.resolve() != tok_dst.resolve():
                shutil.copyfile(tok_src, tok_dst)
        log = JsonlLogger(train_cfg.log_file) if is_master else None

        raw_model = Transformer(model_cfg).to(device)
        opt = torch.optim.AdamW(
            decay_param_groups(raw_model, train_cfg.weight_decay, train_cfg.lr, train_cfg.betas),
            lr=train_cfg.lr, betas=tuple(train_cfg.betas), fused=is_cuda)

        start_step, best_val = 0, float("inf")
        if sum(bool(x) for x in (resume, upcycle_from, init_from)) > 1:
            raise ValueError("set at most one of resume, upcycle_from, or init_from")
        if init_from:
            state = torch.load(init_from, map_location=device, weights_only=False)
            raw_model.load_state_dict(state["model"], strict=True)
            if is_master and not quiet:
                print(f"[kuon] continual pretraining {name} initialized from {init_from} "
                      f"(source step {state.get('step', 0)})")
        if upcycle_from:
            state = torch.load(upcycle_from, map_location=device, weights_only=False)
            missing, unexpected = raw_model.load_state_dict(state["model"], strict=False)
            if unexpected or any(not k.startswith("loop_gate.") for k in missing):
                raise ValueError(f"upcycle_from mismatch: missing={missing}, unexpected={unexpected}")
            if is_master and not quiet:
                print(f"[kuon] upcycled {name} from dense checkpoint {upcycle_from} "
                      f"(source step {state.get('step', 0)})")
        if resume:
            state = torch.load(resume, map_location=device, weights_only=False)
            raw_model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            start_step = state["step"]
            if not bilingual or state["train_cfg"]["data_dir"] == data_dir:
                best_val = state.get("val_loss", float("inf"))
            if is_master and not quiet:
                print(f"[kuon] resumed {name} from {resume} (step {start_step})")

        ddp_model = DDP(raw_model, device_ids=[local_rank]) if ddp else None
        model = ddp_model if ddp_model is not None else raw_model
        if compile_ and is_cuda:
            model = torch.compile(model)

        block, bs, accum = model_cfg.block_size, train_cfg.batch_size, train_cfg.grad_accum
        tokens_per_step = block * bs * accum * world_size
        if is_master and not quiet:
            ddp_note = f", world_size={world_size}" if ddp else ""
            print(f"[kuon] {name}: {param_count(raw_model)/1e6:.2f}M params, "
                  f"device={device}{ddp_note}, steps={steps}, "
                  f"tokens/step={tokens_per_step:,}")

        end_step = min(steps, stop_step) if stop_step is not None else steps
        if not 0 <= schedule_start <= start_step < end_step <= steps:
            raise ValueError("expected schedule_start <= checkpoint step < end_step <= target steps")
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # mixed precision: native hardware bf16 only on Ampere+ (SM >= 8.0);
        # Turing/Volta (T4/P100 on Kaggle/Colab) MUST use fp16 + GradScaler —
        # PyTorch's is_bf16_supported() defaults to including_emulation=True,
        # which silently disables Tensor Cores on T4 (~6-10x slower).
        hw_bf16 = False
        if is_cuda and torch.cuda.get_device_capability()[0] >= 8:
            try:
                hw_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                hw_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = (torch.bfloat16 if hw_bf16 else torch.float16) if is_cuda else None
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        if start_step > 0:
            # Advance RNG on resume so resumed runs draw fresh data crops instead of
            # replaying step-0..N batches from the initial seed and overfitting them.
            seed_all(train_cfg.seed + start_step * 9973 + rank * 10007)
        train_rng = (
            np.random.default_rng(train_cfg.seed + start_step * 9973 + rank * 10007)
            if (ddp or start_step > 0)
            else None
        )

        def validate():
            losses = {lang: estimate_val_loss(raw_model, data, block, bs, device,
                                              train_cfg.eval_batches, amp_dtype=amp_dtype)
                      for lang, data in val_data.items()}
            combined = 0.6 * losses["en"] + 0.4 * losses["id"] if bilingual else losses[""]
            detail = {f"val_loss_{lang}": round(losses[lang], 4) for lang in ("en", "id")} if bilingual else {}
            return combined, detail

        if bilingual:
            if is_master:
                base_val, detail = validate()
                print(f"[kuon] baseline step {start_step} | val EN {detail['val_loss_en']:.4f} "
                      f"| val ID {detail['val_loss_id']:.4f}", flush=True)
                log.write(name=name, step=start_step, val_loss=round(base_val, 4),
                          elapsed=0.0, **detail)
            if ddp:
                dist.barrier()

        model.train()
        running: float | None = None
        first_loss: float | None = None
        t_start, t_iter = time.time(), time.time()
        previous_eval = start_step

        for step in range(start_step, end_step):
            lr = get_lr(step - schedule_start, train_cfg.lr, train_cfg.min_lr,
                        train_cfg.warmup_steps, steps - schedule_start)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            dloss_t = torch.zeros((), device=device, dtype=torch.float32)
            for microstep in range(accum):
                x, y = get_batch(train_data[cycle[microstep % len(cycle)]], block, bs, device,
                                 rng=train_rng)
                sync_ctx = (ddp_model.no_sync()
                            if ddp_model is not None and microstep < accum - 1
                            else nullcontext())
                with sync_ctx:
                    if amp_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)
                    scaler.scale(loss / accum).backward()
                dloss_t += loss.detach().float()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            step_loss_t = dloss_t / accum
            if ddp:
                dist.all_reduce(step_loss_t, op=dist.ReduceOp.AVG)
            step_loss = step_loss_t.item()
            running = step_loss if running is None else 0.98 * running + 0.02 * step_loss
            if first_loss is None:
                first_loss = step_loss

            if (step + 1 - schedule_start) % train_cfg.eval_interval == 0 or step + 1 == end_step:
                if is_master:
                    dt = time.time() - t_iter
                    t_iter = time.time()
                    interval = step + 1 - previous_eval
                    previous_eval = step + 1
                    tok_s = int(tokens_per_step * interval / max(dt, 1e-9))

                    val, detail = validate()
                    detail_text = "".join(f" | val {lang.upper()} {detail[f'val_loss_{lang}']:.4f}"
                                          for lang in ("en", "id")) if detail else ""
                    elapsed = round(time.time() - t_start, 1)
                    print(f"step {step+1:6d}/{steps} | train {running:.4f} | val {val:.4f}{detail_text} | "
                          f"lr {lr:.2e} | {tok_s:,} tok/s | {elapsed}s", flush=True)
                    log.write(name=name, step=step + 1, train_loss=round(running, 4),
                              val_loss=round(val, 4), lr=lr, tokens_per_sec=tok_s, elapsed=elapsed, **detail)

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
                if ddp:
                    dist.barrier()

        return {
            "name": name,
            "steps": steps,
            "first_loss": first_loss,
            "last_loss": running,
            "best_val": best_val,
            "ckpt": str(ckpt_dir / f"{name}-last.pt"),
        }
    finally:
        del model, ddp_model, raw_model, opt
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

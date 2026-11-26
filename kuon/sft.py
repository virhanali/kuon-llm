"""Supervised fine-tuning (SFT) for Kuon chat models.

Fine-tunes a pretrained checkpoint on `{user, assistant}` JSONL data with
prompt-masked cross-entropy: the model is trained to produce the assistant
turns, not the user prompts.

Checkpoints produced here use the same dict format as the pretrainer, so
`scripts/generate.py` works on them directly.
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

from kuon.chat import build_example, format_prompt
from kuon.config import ModelConfig
from kuon.generate import generate
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer
from kuon.utils import seed_all


def load_pairs(path: str | Path) -> list[dict]:
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


def _build_pair(tokenizer, p: dict, block: int, default_business: bool):
    if p.get("raw_text"):
        ids = tokenizer.encode(p["text"], add_eos=True)
        if len(ids) < 8:
            return [], []
        ids = ids[:block + 1]
        return ids[:-1], ids[1:]
    return build_example(
        tokenizer, p["user"], p["assistant"], block,
        evidence=p.get("evidence"),
        business=bool(p.get("business", default_business)),
    )


def _collate_batch(picks, pad_id: int, device: str):
    length = max(len(x) for x, _ in picks)
    x = torch.tensor([ids + [pad_id] * (length - len(ids)) for ids, _ in picks], dtype=torch.long)
    y = torch.tensor([lbl + [-100] * (length - len(lbl)) for _, lbl in picks], dtype=torch.long)
    return x.to(device), y.to(device)


class _EpochSampler:
    """Shuffled without-replacement batch sampler so every SFT row gets balanced updates."""

    def __init__(self, examples: list, seed: int = 0):
        self.examples = list(examples)
        self.rng = random.Random(seed)
        self._order: list[int] = []
        self._pos = 0

    def next_batch(self, batch_size: int) -> list:
        picks = []
        while len(picks) < batch_size:
            if self._pos >= len(self._order):
                self._order = list(range(len(self.examples)))
                self.rng.shuffle(self._order)
                self._pos = 0
            picks.append(self.examples[self._order[self._pos]])
            self._pos += 1
        return picks


@torch.no_grad()
def _masked_loss(model, examples, batch_size: int, pad_id: int, device: str,
                 amp_dtype: torch.dtype | None = None) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        length = max(len(x) for x, _ in chunk)
        x = torch.tensor([ids + [pad_id] * (length - len(ids)) for ids, _ in chunk], dtype=torch.long).to(device)
        y = torch.tensor([lbl + [-100] * (length - len(lbl)) for _, lbl in chunk], dtype=torch.long).to(device)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(x)
                loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                           ignore_index=-100, reduction="sum")
        else:
            logits, _ = model(x)
            loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                       ignore_index=-100, reduction="sum")
        total += loss_sum.item()
        count += (y != -100).sum().item()
    model.train()
    return total / max(count, 1)


def sft(base_ckpt: str, data_dir: str, *, out_path: str | None = None, steps: int = 300,
        batch_size: int = 8, grad_accum: int = 1, lr: float = 2e-4, min_lr: float = 0.0,
        warmup_steps: int = 0, weight_decay: float = 0.01,
        eval_interval: int = 0, patience: int = 0,
        block_size: int | None = None, device: str | None = None, seed: int = 0,
        sample_prompt: str = "Tell me a story.", quiet: bool = False,
        tokenizer_path: str | Path | None = None, business: bool = False,
        grad_checkpoint: bool = False) -> dict:
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
    train_ex = [_build_pair(tokenizer, p, block, business)
                for p in load_pairs(data / "sft_train.jsonl")]
    val_ex = [_build_pair(tokenizer, p, block, business)
              for p in load_pairs(data / "sft_val.jsonl")]
    train_ex = [(x, y) for x, y in train_ex if any(l != -100 for l in y)]
    val_ex = [(x, y) for x, y in val_ex if any(l != -100 for l in y)]
    assert train_ex, "no usable SFT examples (assistant text empty?)"

    is_cuda = device.startswith("cuda")
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(state["model"])
    model.grad_checkpoint = bool(grad_checkpoint)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=is_cuda,
    )
    hw_bf16 = False
    if is_cuda and torch.cuda.get_device_capability()[0] >= 8:
        try:
            hw_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            hw_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = (torch.bfloat16 if hw_bf16 else torch.float16) if is_cuda else None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    if not quiet:
        print(f"[sft] base={base_ckpt} | {len(train_ex)} train pairs, {len(val_ex)} val pairs "
              f"| device={device} | steps={steps} bs={batch_size}x{grad_accum} lr={lr:.1e} "
              f"wd={weight_decay} warmup={warmup_steps} eval_int={eval_interval}", flush=True)

    sampler = _EpochSampler(train_ex, seed=seed)
    first_loss = None
    running = None
    best_val_loss = float("inf")
    best_step = 0
    best_state = None
    bad_evals = 0
    if eval_interval > 0 and val_ex:
        best_val_loss = _masked_loss(model, val_ex, batch_size, pad_id, device, amp_dtype=amp_dtype)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not quiet:
            print(f"[sft] baseline step 0 | val {best_val_loss:.4f}", flush=True)

    model.train()
    start_time = time.time()
    steps_run = 0
    for step in range(steps):
        steps_run = step + 1
        if warmup_steps > 0 and step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = _collate_batch(sampler.next_batch(batch_size), pad_id, device)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            else:
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            scaler.scale(loss / grad_accum).backward()
            step_loss += loss.item() / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        first_loss = step_loss if first_loss is None else first_loss
        running = step_loss if running is None else 0.95 * running + 0.05 * step_loss

        if eval_interval > 0 and val_ex and (step + 1) % eval_interval == 0:
            cur_val = _masked_loss(model, val_ex, batch_size, pad_id, device, amp_dtype=amp_dtype)
            improved = cur_val < best_val_loss
            if improved:
                best_val_loss = cur_val
                best_step = step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_evals = 0
            else:
                bad_evals += 1
            if not quiet:
                tag = " *" if improved else ""
                print(f"step {step+1:4d}/{steps} | loss {running:.4f} | val {cur_val:.4f}{tag} | "
                      f"lr {cur_lr:.2e} | {time.time() - start_time:.0f}s", flush=True)
            if patience > 0 and bad_evals >= patience:
                if not quiet:
                    print(f"[sft] early stopping at step {step+1} (best step {best_step}, "
                          f"best val {best_val_loss:.4f})", flush=True)
                break
        elif not quiet and (step + 1) % max(1, steps // 4) == 0:
            print(f"step {step+1:4d}/{steps} | loss {running:.4f} | lr {cur_lr:.2e} | "
                  f"{time.time() - start_time:.0f}s", flush=True)

    final_val = _masked_loss(model, val_ex, batch_size, pad_id, device, amp_dtype=amp_dtype) if val_ex else float("nan")
    if eval_interval > 0 and best_state is not None:
        if final_val < best_val_loss:
            best_val_loss = final_val
            best_step = steps_run
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        val_loss = best_val_loss
    else:
        val_loss = final_val
        best_val_loss = final_val
        best_step = steps_run

    out = Path(out_path or f"checkpoints/{data.parent.name}-sft.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": state["train_cfg"],
        "step": state.get("step", 0),
        "val_loss": val_loss,
        "sft_meta": {"best_step": best_step, "steps_run": steps_run, "final_val_loss": final_val},
    }, tmp)
    os.replace(tmp, out)
    dest_tok = out.parent / "tokenizer.json"
    if Path(tokenizer_path).resolve() != dest_tok.resolve():
        shutil.copyfile(tokenizer_path, dest_tok)

    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(format_prompt(sample_prompt, business=business))],
                              dtype=torch.long, device=device)
    sample_ids = [n[0] for n in generate(model, prompt_ids, max_new_tokens=80,
                                         temperature=0.7, eos_id=tokenizer.eos_id)]
    sample = tokenizer.decode(sample_ids)
    if not quiet:
        print(f"[sft] done | best_step={best_step} val {val_loss:.4f} (final {final_val:.4f}) | saved {out}", flush=True)
        print(f"  sample> {sample.strip()[:200]}", flush=True)

    return {
        "first_loss": first_loss,
        "last_loss": running,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "sample": sample,
        "ckpt": str(out),
        "steps": steps_run,
    }

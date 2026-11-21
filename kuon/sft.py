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
import random
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


def _collate(examples, batch_size: int, pad_id: int, device: str):
    picks = [random.choice(examples) for _ in range(batch_size)]
    length = max(len(x) for x, _ in picks)
    x = torch.tensor([ids + [pad_id] * (length - len(ids)) for ids, _ in picks], dtype=torch.long)
    y = torch.tensor([lbl + [-100] * (length - len(lbl)) for _, lbl in picks], dtype=torch.long)
    return x.to(device), y.to(device)


@torch.no_grad()
def _masked_loss(model, examples, batch_size: int, pad_id: int, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        length = max(len(x) for x, _ in chunk)
        x = torch.tensor([ids + [pad_id] * (length - len(ids)) for ids, _ in chunk], dtype=torch.long).to(device)
        y = torch.tensor([lbl + [-100] * (length - len(lbl)) for _, lbl in chunk], dtype=torch.long).to(device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                 ignore_index=-100, reduction="sum").item()
        count += (y != -100).sum().item()
    model.train()
    return total / max(count, 1)


def sft(base_ckpt: str, data_dir: str, *, out_path: str | None = None, steps: int = 300,
        batch_size: int = 8, lr: float = 2e-4, block_size: int | None = None,
        device: str | None = None, seed: int = 0,
        sample_prompt: str = "Tell me a story.", quiet: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)

    state = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**state["model_cfg"])
    tokenizer = BPETokenizer.load(Path(state["train_cfg"]["data_dir"]) / "tokenizer.json")
    block = block_size or model_cfg.block_size
    pad_id = tokenizer.eos_id if tokenizer.eos_id is not None else 0

    data = Path(data_dir)
    train_ex = [build_example(tokenizer, p["user"], p["assistant"], block)
                for p in load_pairs(data / "sft_train.jsonl")]
    val_ex = [build_example(tokenizer, p["user"], p["assistant"], block)
              for p in load_pairs(data / "sft_val.jsonl")]
    train_ex = [(x, y) for x, y in train_ex if any(l != -100 for l in y)]
    val_ex = [(x, y) for x, y in val_ex if any(l != -100 for l in y)]
    assert train_ex, "no usable SFT examples (assistant text empty?)"

    model = Transformer(model_cfg).to(device)
    model.load_state_dict(state["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    if not quiet:
        print(f"[sft] base={base_ckpt} | {len(train_ex)} train pairs, {len(val_ex)} val pairs "
              f"| device={device} | steps={steps}")

    first_loss = None
    running = None
    model.train()
    start_time = time.time()
    for step in range(steps):
        progress = step / max(1, steps)
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y = _collate(train_ex, batch_size, pad_id, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        value = loss.item()
        first_loss = value if first_loss is None else first_loss
        running = value if running is None else 0.95 * running + 0.05 * value

        if not quiet and (step + 1) % max(1, steps // 4) == 0:
            print(f"step {step+1:4d}/{steps} | loss {running:.4f} | lr {cur_lr:.2e} | "
                  f"{time.time() - start_time:.0f}s", flush=True)

    val_loss = _masked_loss(model, val_ex, batch_size, pad_id, device) if val_ex else float("nan")

    out = Path(out_path or f"checkpoints/{data.parent.name}-sft.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": state["train_cfg"],
        "step": state.get("step", 0),
        "val_loss": val_loss,
    }, out)

    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(format_prompt(sample_prompt))],
                              dtype=torch.long, device=device)
    sample_ids = [n[0] for n in generate(model, prompt_ids, max_new_tokens=80,
                                         temperature=0.7, eos_id=tokenizer.eos_id)]
    sample = tokenizer.decode(sample_ids)
    if not quiet:
        print(f"[sft] done | val {val_loss:.4f} | saved {out}")
        print(f"  sample> {sample.strip()[:200]}")

    return {
        "first_loss": first_loss,
        "last_loss": running,
        "val_loss": val_loss,
        "sample": sample,
        "ckpt": str(out),
        "steps": steps,
    }

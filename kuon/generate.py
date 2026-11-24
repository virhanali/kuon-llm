"""Sampling and streaming generation.

`generate()` is a generator that yields new token ids one decoding step
at a time, so callers can stream text as it is produced. It uses the
model's KV cache: the prompt is prefilled once, then each new token is a
single forward pass. When the cache fills up (context == block_size),
the window slides and the cache is rebuilt from the last block_size
tokens — the simplest correct strategy for a fixed context window.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kuon.model import KVCache


def sample_from_logits(logits: torch.Tensor, temperature: float = 0.8,
                       top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    """Sample one token id per row of `logits` (B, vocab).

    temperature <= 0 means greedy decoding (argmax).
    top_k keeps only the k most likely tokens; top_p (nucleus) keeps the
    smallest set whose cumulative probability exceeds p. Both may be
    disabled by passing 0 / 1.0."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits.float() / max(temperature, 1e-8)

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # drop tokens whose cumulative mass (excluding the token itself)
        # already exceeds top_p — the first token past the threshold is kept
        drop = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def _apply_repetition_controls(
    logits: torch.Tensor,
    generated: list[list[int]],
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Apply Keskar-style repetition penalty and n-gram blocking on generated tokens only."""
    if repetition_penalty == 1.0 and no_repeat_ngram_size <= 0:
        return logits
    out = logits.clone()
    for row, seq in enumerate(generated):
        if not seq:
            continue
        if repetition_penalty > 1.0:
            for tid in set(seq):
                if out[row, tid] > 0:
                    out[row, tid] /= repetition_penalty
                else:
                    out[row, tid] *= repetition_penalty
        if no_repeat_ngram_size >= 2 and len(seq) >= no_repeat_ngram_size - 1:
            n = no_repeat_ngram_size
            prefix = tuple(seq[-(n - 1):])
            banned: set[int] = set()
            for idx in range(len(seq) - n + 1):
                if tuple(seq[idx:idx + n - 1]) == prefix:
                    banned.add(seq[idx + n - 1])
            for tid in banned:
                out[row, tid] = float("-inf")
    return out


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
             eos_id: int | None = None, repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 0):
    """Yield lists of new token ids (one per batch row) for up to
    max_new_tokens. Stops early when every row has produced eos_id."""
    model.eval()
    device = prompt_ids.device
    cfg = model.cfg
    block = cfg.block_size
    dtype = next(model.parameters()).dtype

    def fresh_caches():
        return [KVCache(prompt_ids.size(0), block, cfg.n_kv_head, cfg.head_dim,
                        dtype, device) for _ in range(cfg.n_exec_layers)]

    ids = prompt_ids[:, -block:]
    caches = fresh_caches()
    logits, _ = model(ids, cache=caches)   # prefill the prompt once
    generated: list[list[int]] = [[] for _ in range(prompt_ids.size(0))]

    for _ in range(max_new_tokens):
        step_logits = _apply_repetition_controls(
            logits[:, -1, :], generated, repetition_penalty, no_repeat_ngram_size
        )
        nxt = sample_from_logits(step_logits, temperature, top_k, top_p)  # (B, 1)
        tok = nxt[:, 0].tolist()
        for row, tid in enumerate(tok):
            generated[row].append(tid)
        yield tok
        if eos_id is not None and tok.count(eos_id) == len(tok):
            return

        ids = torch.cat([ids, nxt], dim=1)[:, -block:]
        if caches[0].len + 1 > block:
            # context is full: slide the window by half a block so we don't
            # rebuild a full-block prefill on every subsequent decode token.
            ids = ids[:, -(block // 2):]
            caches = fresh_caches()
            logits, _ = model(ids, cache=caches)
        else:
            logits, _ = model(nxt, cache=caches)   # single-token decode step


def sample_text(model, tokenizer, prompt: str, n_tokens: int = 96,
                temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                device: str = "cpu") -> str:
    """Convenience wrapper used by the trainer to print progress samples."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out: list[int] = []
    for tok in generate(model, ids, max_new_tokens=n_tokens, temperature=temperature,
                        top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id):
        out.append(tok[0])
        if tokenizer.eos_id in tok:
            break
    return tokenizer.decode(out)

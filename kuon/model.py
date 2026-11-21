"""Kuon transformer: a modern, Llama-style decoder-only language model.

Read this file top to bottom; each piece is small and only depends on
the pieces above it:

    RMSNorm               simpler, better LayerNorm (no mean-centering, no bias)
    build_rope_tables     rotary position embeddings, precomputed cos/sin
    apply_rope            rotates q and k so attention depends on relative
                          positions only
    KVCache               preallocated per-layer key/value buffers
    CausalSelfAttention   multi-head attention with grouped-query attention
                          (GQA), RoPE, and an optional KV cache
    SwiGLU                gated feed-forward: down(silu(gate(x)) * up(x))
    Block                 x += attn(norm(x)); x += ffn(norm(x))   (pre-norm)
    Transformer           embeddings -> blocks -> final norm -> LM head

Design choices (all standard in modern open models):
  * pre-norm residual blocks -> stable training without warmup tricks
  * RoPE instead of learned absolute positions -> generalizes past
    anything seen at train time (up to block_size)
  * GQA: fewer key/value heads than query heads -> smaller KV cache
    at inference, same accuracy at this scale
  * tied embeddings: lm_head shares the token embedding matrix -> saves
    vocab_size * n_embd parameters, which matters a lot for small models
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kuon.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization: x / sqrt(mean(x^2) + eps) * weight."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_rope_tables(block_size: int, head_dim: int, theta: float = 10000.0):
    """Precompute cos/sin tables for rotary embeddings.

    Returns two (block_size, head_dim) float32 tensors. Row t holds the
    rotation angles for absolute position t; each half of the head
    dimension rotates at frequencies theta^(-2i/head_dim).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(block_size, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                # (block, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)         # duplicate halves (NeoX style)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Rotate `x` (B, n_head, T, head_dim) by the angles for positions
    [offset, offset+T). Standard NeoX/Llama formulation:

        out = x * cos + rotate_half(x) * sin

    with rotate_half(x) = concat(-x2, x1); this is exactly a 2-D rotation
    of each (i, i + head_dim/2) channel pair, so attention scores end up
    depending only on relative distance between positions."""
    T = x.shape[-2]
    c = cos[offset:offset + T].to(x.dtype)
    s = sin[offset:offset + T].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * c + rotated * s


class KVCache:
    """Preallocated key/value storage for one attention layer.

    Keys/values for processed positions are written at [len, len+T) and
    attention reads the prefix [:len+T]. The buffer is transposed to
    (batch, n_kv_head, max_len, head_dim) so slicing feeds SDPA directly.
    """

    def __init__(self, batch_size: int, max_len: int, n_kv_head: int, head_dim: int,
                 dtype: torch.dtype | None = None, device: torch.device | None = None):
        self.k = torch.zeros(batch_size, n_kv_head, max_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.len = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, C = x.shape
        nh, nkv, hd = self.n_head, self.n_kv_head, self.head_dim

        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, nkv, hd).transpose(1, 2)  # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, nkv, hd).transpose(1, 2)

        # rotate queries and keys by their absolute positions
        offset = cache.len if cache is not None else 0
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            past = cache.len
            cache.k[:, :, past:past + T] = k
            cache.v[:, :, past:past + T] = v
            cache.len = past + T
            k = cache.k[:, :, :cache.len]
            v = cache.v[:, :, :cache.len]

        # grouped-query attention: replicate k/v heads to match query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if cache is None or cache.len == T:
            # training / fresh prefill: queries and keys have the same length
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0)
        elif T == 1:
            # single-token decode: every key is in the past, nothing to mask
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # mid-cache prefill (chunk): explicit causal mask
            past = cache.len - T
            mask = torch.ones(T, cache.len, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))


class SwiGLU(nn.Module):
    """Gated feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        x = x + self.attn(self.norm_attn(x), cos, sin, cache)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_tables(cfg.block_size, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # execution schedule over stored layers; looping repeats the middle
        # block range without adding parameters (recurrent depth)
        if cfg.looping:
            schedule = list(range(0, cfg.loop_start))
            for _ in range(cfg.loop_passes):
                schedule.extend(range(cfg.loop_start, cfg.loop_end))
            schedule.extend(range(cfg.loop_end, cfg.n_layer))
        else:
            schedule = list(range(cfg.n_layer))
        self._schedule = schedule

        # weight init: N(0, 0.02) everywhere, except residual-out projections
        # which get N(0, 0.02/sqrt(2*n_layer)) to keep the residual stream's
        # variance from growing with depth (GPT-2/nanoGPT convention).
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_init)
        scaled = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=scaled)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: list[KVCache] | None = None):
        """idx: (B, T) int64 tokens, T <= block_size.
        With cache=None this is a training/eval forward over a full window.
        With a list of per-layer KVCache objects it is a generation step
        (prefill or single-token decode) that reuses past keys/values.

        Returns (logits (B, T, vocab), loss) where loss is the mean
        next-token cross-entropy when `targets` is given."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"context {T} > block_size {self.cfg.block_size}"

        x = self.drop(self.embed(idx))
        if cache is not None and self.cfg.looping:
            raise NotImplementedError(
                "KV cache is not supported for looped models; "
                "call generate() which falls back to no-cache decoding")
        for layer_idx in self._schedule:
            block = self.blocks[layer_idx]
            x = block(x, self.rope_cos, self.rope_sin,
                      cache[layer_idx] if cache is not None else None)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

"""Tests for the transformer: shapes, causality, RoPE, GQA, KV cache."""

import math

import torch
import torch.nn.functional as F

from kuon.config import ModelConfig
from kuon.generate import generate
from kuon.model import Transformer, apply_rope, build_rope_tables


def tiny_cfg(**over) -> ModelConfig:
    defaults = dict(vocab_size=97, block_size=64, n_layer=2, n_embd=64,
                    n_head=2, n_kv_head=1, ffn_hidden=96, dropout=0.0)
    defaults.update(over)
    return ModelConfig(**defaults)


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = Transformer(tiny_cfg())
    idx = torch.randint(0, 97, (2, 32))
    logits, loss = model(idx)
    assert logits.shape == (2, 32, 97)
    assert loss is None

    targets = torch.randint(0, 97, (2, 32))
    _, loss = model(idx, targets)
    # an untrained model should be near-uniform: loss ~ ln(vocab_size)
    assert 3.5 < loss.item() < 5.5
    loss.backward()
    assert model.embed.weight.grad is not None
    assert model.blocks[0].attn.q_proj.weight.grad is not None


def test_causality():
    """Changing future tokens must not change past logits."""
    torch.manual_seed(0)
    model = Transformer(tiny_cfg()).eval()
    idx1 = torch.randint(0, 97, (2, 40))
    idx2 = idx1.clone()
    idx2[:, 25:] = torch.randint(0, 97, (2, 15))  # perturb ONLY the future
    with torch.no_grad():
        l1, _ = model(idx1)
        l2, _ = model(idx2)
    torch.testing.assert_close(l1[:, :25], l2[:, :25], rtol=1e-5, atol=1e-5)


def test_tied_weights():
    model = Transformer(tiny_cfg())
    assert model.lm_head.weight is model.embed.weight


def test_rope_relative_positions():
    """Attention scores under RoPE depend only on relative distance."""
    cos, sin = build_rope_tables(64, 16, 10000.0)
    vq, vk = torch.randn(16), torch.randn(16)
    q = vq.view(1, 1, 1, 16).expand(1, 1, 40, 16).contiguous()
    k = vk.view(1, 1, 1, 16).expand(1, 1, 40, 16).contiguous()
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    for i, j in [(3, 1), (20, 4), (30, 25)]:
        s = (qr[0, 0, i] * kr[0, 0, j]).sum()
        s_shifted = (qr[0, 0, i + 5] * kr[0, 0, j + 5]).sum()
        torch.testing.assert_close(s, s_shifted, rtol=1e-4, atol=1e-5)


def test_gqa_matches_reference():
    """The GQA + RoPE + SDPA path must equal plain manual attention."""
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=1, n_head=2, n_kv_head=1, n_embd=32,
                   ffn_hidden=48, block_size=16)
    model = Transformer(cfg).eval()
    attn = model.blocks[0].attn

    B, T, C = 1, 12, cfg.n_embd
    x = torch.randn(B, T, C)
    nh, nkv, hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim

    with torch.no_grad():
        y = attn(x, model.rope_cos, model.rope_sin)

        q = attn.q_proj(x).view(B, T, nh, hd).permute(0, 2, 1, 3)
        k = attn.k_proj(x).view(B, T, nkv, hd).permute(0, 2, 1, 3)
        v = attn.v_proj(x).view(B, T, nkv, hd).permute(0, 2, 1, 3)
        q = apply_rope(q, model.rope_cos, model.rope_sin)
        k = apply_rope(k, model.rope_cos, model.rope_sin)
        k = k.repeat_interleave(nh // nkv, dim=1)
        v = v.repeat_interleave(nh // nkv, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        y_ref = attn.o_proj((att @ v).permute(0, 2, 1, 3).reshape(B, T, C))

    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-5)


def test_kv_cache_greedy_matches_no_cache():
    """Cached decoding must produce identical tokens to full re-forwards."""
    torch.manual_seed(0)
    model = Transformer(tiny_cfg()).eval()
    prompt = torch.randint(0, 97, (1, 10))
    n_new = 12

    # no cache: recompute the full sequence every step
    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits, _ = model(ids)
            nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)

    # with cache: prefill once, then single-token steps
    out = []
    for new in generate(model, prompt, n_new, temperature=0):
        out.append(new[0])
    assert out == ids[0, 10:].tolist()


def test_looped_schedule_and_param_count():
    """Looping repeats blocks without adding parameters."""
    plain = Transformer(tiny_cfg(n_layer=3))
    looped = Transformer(tiny_cfg(n_layer=3, loop_start=1, loop_end=2, loop_passes=2))
    assert plain._schedule == [0, 1, 2]
    assert looped._schedule == [0, 1, 1, 2]
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in looped.parameters())


def test_looped_forward_backward():
    torch.manual_seed(0)
    model = Transformer(tiny_cfg(n_layer=3, loop_start=0, loop_end=3, loop_passes=3))
    idx = torch.randint(0, 97, (2, 16))
    targets = torch.randint(0, 97, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 97)
    loss.backward()
    assert model.blocks[1].ffn.gate_proj.weight.grad is not None


def test_looped_generation_no_cache():
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=3, loop_start=1, loop_end=2, loop_passes=2, block_size=32)
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, 97, (1, 8))
    ids = [new[0] for new in generate(model, prompt, max_new_tokens=5, temperature=0)]
    assert len(ids) == 5
    assert all(0 <= i < 97 for i in ids)


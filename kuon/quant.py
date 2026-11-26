"""Portable per-channel symmetric INT8 weight quantization and CPU runtime acceleration.

Stores Kuon checkpoints at ~3.7x smaller disk size (~315 MB for Kuon-2 300M vs ~1.18 GB FP32)
using standard PyTorch int8 + fp16 tensors (zero opaque backend blobs, portable across PyTorch
versions and CPU/GPU architectures). On CPU load, optionally accelerates Linear layers with
PyTorch dynamic INT8 kernels (~2.2x faster decode on 4 vCPUs) with graceful fallback.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

QUANT_FORMAT_INT8 = "int8_per_channel"


def quantize_state_dict_int8(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, int]]:
    """Quantize 2D projection weights to per-channel symmetric int8 + fp16 scale.

    Embeddings stay fp16 (to preserve tied lm_head quality and lookup speed) and 1D norms
    stay fp32 (tiny footprint, sensitive to precision).
    """
    q_weights: dict[str, torch.Tensor] = {}
    q_scales: dict[str, torch.Tensor] = {}
    fp32_bytes = 0
    quant_bytes = 0

    has_embed = "embed.weight" in state_dict
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        fp32_bytes += t.numel() * 4
        # Skip duplicate tied lm_head.weight when embed.weight is present
        if name == "lm_head.weight" and has_embed and t.shape == state_dict["embed.weight"].shape:
            if torch.equal(t, state_dict["embed.weight"].detach().cpu()):
                continue
        if t.dim() == 2 and name != "embed.weight" and t.shape[1] >= 16:
            tf = t.float()
            max_abs = tf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale = (max_abs / 127.0).to(torch.float16)
            qw = torch.round(tf / scale.float()).clamp(-127, 127).to(torch.int8)
            q_weights[name] = qw
            q_scales[name] = scale
            quant_bytes += qw.numel() * qw.element_size() + scale.numel() * scale.element_size()
        elif name == "embed.weight":
            te = t.to(torch.float16)
            q_weights[name] = te
            quant_bytes += te.numel() * te.element_size()
        else:
            tf = t.to(torch.float32)
            q_weights[name] = tf
            quant_bytes += tf.numel() * tf.element_size()

    payload = {"weights": q_weights, "scales": q_scales}
    stats = {
        "fp32_bytes": fp32_bytes,
        "quant_bytes": quant_bytes,
        "quantized_tensors": len(q_scales),
    }
    return payload, stats


def dequantize_state_dict_int8(
    payload: dict[str, Any],
    *,
    tie_word_embeddings: bool = True,
) -> dict[str, torch.Tensor]:
    """Reconstruct an FP32 state_dict from a portable int8_per_channel payload."""
    weights: dict[str, torch.Tensor] = payload["weights"]
    scales: dict[str, torch.Tensor] = payload["scales"]
    out: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        if name in scales:
            out[name] = tensor.float() * scales[name].float()
        else:
            out[name] = tensor.float()
    if tie_word_embeddings and "embed.weight" in out and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["embed.weight"]
    return out


def apply_cpu_int8_runtime(model: nn.Module) -> nn.Module:
    """Apply selective per-channel dynamic INT8 on transformer blocks (keeping lm_head & loop_gate FP32)."""
    try:
        from torch.ao.quantization import PerChannelMinMaxObserver, PlaceholderObserver, QConfig

        pc_qconfig = QConfig(
            activation=PlaceholderObserver.with_args(dtype=torch.quint8, is_dynamic=True),
            weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = getattr(model, "blocks", None)
            if isinstance(blocks, nn.ModuleList):
                for i, blk in enumerate(blocks):
                    blocks[i] = torch.ao.quantization.quantize_dynamic(
                        blk, {nn.Linear: pc_qconfig}, dtype=torch.qint8
                    )
                return model
            return torch.ao.quantization.quantize_dynamic(
                model, {nn.Linear: pc_qconfig}, dtype=torch.qint8
            )
    except Exception:
        return model

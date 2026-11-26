"""Pluggable answer engines for the agent.

All engines implement a single method: `complete(prompt) -> str`.

- MockEngine      deterministic, for tests and dry runs (no model needed)
- KuonEngine      a trained Kuon checkpoint via the kuon package (torch)
- OllamaEngine    a local model served by Ollama (e.g. qwen2.5-coder:7b)
- LlamaCppEngine  a GGUF file via the llama-cli binary

Swapping engines changes quality; the agent loop stays identical.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Protocol


class Engine(Protocol):
    name: str

    def complete(self, prompt: str, max_tokens: int = 256) -> str:  # pragma: no cover - protocol
        ...


class MockEngine:
    """Deterministic engine for tests: echoes a grounded-looking stub."""

    name = "mock"

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        return "[mock] Based on the evidence provided, I cannot verify this answer."


class KuonEngine:
    name = "kuon"

    def __init__(self, checkpoint: str, device: str | None = None,
                 *, tokenizer_path: str | Path | None = None, temperature: float = 0.0,
                 repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 4,
                 quantize_cpu: bool | None = None):
        import os
        import torch

        from kuon.config import ModelConfig
        from kuon.model import Transformer
        from kuon.quant import QUANT_FORMAT_INT8, apply_cpu_int8_runtime, dequantize_state_dict_int8
        from kuon.tokenizer import BPETokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            cpu_threads = os.environ.get("KUON_CPU_THREADS")
            if cpu_threads and cpu_threads.isdigit():
                torch.set_num_threads(int(cpu_threads))
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model_cfg = ModelConfig(**state["model_cfg"])
        # the tokenizer lives next to the checkpoint when shipped (release
        # asset), or in the original data dir on the training machine
        candidates = [Path(tokenizer_path)] if tokenizer_path else [
            Path(checkpoint).parent / "tokenizer.json",
            Path(state.get("train_cfg", {}).get("data_dir", "")) / "tokenizer.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                self.tokenizer = BPETokenizer.load(candidate)
                break
        else:
            raise FileNotFoundError(
                "tokenizer.json not found next to the checkpoint. "
                f"Looked in: {[str(c) for c in candidates]}. "
                "Download it from the GitHub release into the checkpoint folder.")
        if self.tokenizer.vocab_size != self.model_cfg.vocab_size:
            raise ValueError("tokenizer vocabulary size does not match checkpoint")
        self.model = Transformer(self.model_cfg).eval()
        is_quant_ckpt = state.get("quant") == QUANT_FORMAT_INT8
        if is_quant_ckpt:
            sd = dequantize_state_dict_int8(
                state["model_quant"], tie_word_embeddings=self.model_cfg.tie_word_embeddings
            )
            self.model.load_state_dict(sd)
        else:
            self.model.load_state_dict(state["model"])
        env_quant = os.environ.get("KUON_QUANT", "").strip().lower()
        if quantize_cpu is not None:
            use_cpu_int8 = bool(quantize_cpu)
        elif env_quant in ("0", "false", "off", "fp32", "none"):
            use_cpu_int8 = False
        else:
            use_cpu_int8 = env_quant in ("dyn_int8", "dynamic", "int8_dyn", "int8", "1", "true")
        self.quant_mode = (
            "int8" if (self.device == "cpu" and use_cpu_int8)
            else ("int8_weights" if is_quant_ckpt else "fp32")
        )
        if self.device == "cpu" and use_cpu_int8:
            self.model = apply_cpu_int8_runtime(self.model)
        else:
            self.model = self.model.to(self.device)

    def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
        return len(self.tokenizer.encode(prompt)) + max_tokens <= self.model_cfg.block_size

    def complete_stream(self, prompt: str, max_tokens: int = 256, *,
                        temperature: float | None = None):
        """Yield incremental decoded text deltas live as each token is generated."""
        from kuon.generate import generate

        temp = self.temperature if temperature is None else float(temperature)
        ids = self.torch.tensor([self.tokenizer.encode(prompt)], dtype=self.torch.long,
                                device=self.device)
        out: list[int] = []
        emitted = ""
        for step in generate(
            self.model, ids, max_new_tokens=max_tokens, temperature=temp,
            top_k=40, top_p=0.9, eos_id=self.tokenizer.eos_id,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        ):
            tid = step[0]
            if self.tokenizer.eos_id is not None and tid == self.tokenizer.eos_id:
                break
            out.append(tid)
            decoded = self.tokenizer.decode(out)
            # Hold back trailing replacement chars from partial multi-byte UTF-8 sequences
            safe = decoded.rstrip("\ufffd")
            if len(safe) > len(emitted):
                delta = safe[len(emitted):]
                emitted = safe
                yield delta
        final = self.tokenizer.decode(out)
        if len(final) > len(emitted):
            yield final[len(emitted):]

    def complete(self, prompt: str, max_tokens: int = 256, *,
                 temperature: float | None = None) -> str:
        from kuon.generate import generate

        temp = self.temperature if temperature is None else float(temperature)
        ids = self.torch.tensor([self.tokenizer.encode(prompt)], dtype=self.torch.long,
                                device=self.device)
        out = [step[0] for step in generate(
            self.model, ids, max_new_tokens=max_tokens, temperature=temp,
            top_k=40, top_p=0.9, eos_id=self.tokenizer.eos_id,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size)]
        if self.tokenizer.eos_id is not None:
            out = [token for token in out if token != self.tokenizer.eos_id]
        return self.tokenizer.decode(out).strip()


class OllamaEngine:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read()).get("response", "").strip()


class LlamaCppEngine:
    name = "llamacpp"

    def __init__(self, model_path: str, binary: str = "llama-cli", n_ctx: int = 4096):
        self.model_path = model_path
        self.binary = binary
        self.n_ctx = n_ctx

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        cmd = [
            self.binary, "-m", self.model_path, "-p", prompt,
            "-n", str(max_tokens), "--temp", "0.3", "-c", str(self.n_ctx),
            "--no-display-prompt",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return result.stdout.strip()

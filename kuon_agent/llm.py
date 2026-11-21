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

    def __init__(self, checkpoint: str, device: str | None = None):
        import torch

        from kuon.config import ModelConfig
        from kuon.model import Transformer
        from kuon.tokenizer import BPETokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model_cfg = ModelConfig(**state["model_cfg"])
        tokenizer_path = Path(state["train_cfg"]["data_dir"]) / "tokenizer.json"
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.model = Transformer(self.model_cfg).to(self.device).eval()
        self.model.load_state_dict(state["model"])

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        from kuon.generate import generate

        ids = self.torch.tensor([self.tokenizer.encode(prompt)], dtype=self.torch.long,
                                device=self.device)
        out = [step[0] for step in generate(
            self.model, ids, max_new_tokens=max_tokens, temperature=0.3,
            top_k=40, top_p=0.9, eos_id=self.tokenizer.eos_id)]
        return self.tokenizer.decode(out)


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

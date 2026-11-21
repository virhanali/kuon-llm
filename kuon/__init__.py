"""Kuon: a small, English-first, open-weight LLM built from scratch.

Every component lives in this repository and is meant to be read
end-to-end: the byte-level BPE tokenizer, the Llama-style transformer,
the training loop, and the inference stack.
"""

__version__ = "0.1.0"

from kuon.config import ModelConfig, TrainConfig, load_run_config
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "Transformer",
    "BPETokenizer",
    "load_run_config",
    "__version__",
]

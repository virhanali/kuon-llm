"""From-scratch retrieval: BM25 over local text corpora.

BM25 is the classic keyword-ranking function (the heart of search engines
before neural retrieval). Implemented here directly: term frequencies,
document frequencies, idf, and length normalization — no dependencies.

Good enough for an honest local RAG demo: 'who is MrBeast' retrieves the
MrBeast page because the rare tokens 'mrbeast' dominate the ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from kuon_agent.text import words


@dataclass
class Chunk:
    text: str
    source: str


def chunk_text(text: str, source: str, max_chars: int = 900,
               overlap_chars: int = 120) -> list[Chunk]:
    """Split text into paragraph-aligned chunks of <= max_chars (hard-wrapping huge paragraphs)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    step = max(1, max_chars - max(0, min(overlap_chars, max_chars // 2)))
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(Chunk(current, source))
                current = ""
            for start in range(0, len(para), step):
                part = para[start:start + max_chars].strip()
                if part:
                    chunks.append(Chunk(part, source))
                if start + max_chars >= len(para):
                    break
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(Chunk(current, source))
            tail = current[-overlap_chars:] if overlap_chars else ""
            if tail and len(tail) + len(para) + 2 <= max_chars:
                current = (tail + "\n\n" + para).strip()
            else:
                current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(Chunk(current, source))
    return chunks


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl = 1.0
        self._finalized = False

    # ---------------------------------------------------------------- build

    def add(self, chunk: Chunk) -> None:
        tokens = words(chunk.text)
        if not tokens:
            return
        self.chunks.append(chunk)
        self._tf.append(Counter(tokens))
        self._len.append(len(tokens))
        for term in set(tokens):
            self._df[term] += 1
        self._finalized = False

    def add_text(self, text: str, source: str, **chunk_kwargs) -> None:
        for chunk in chunk_text(text, source, **chunk_kwargs):
            self.add(chunk)

    def add_file(self, path: str, source: str | None = None) -> None:
        from pathlib import Path

        p = Path(path)
        self.add_text(p.read_text(encoding="utf-8", errors="replace"),
                      source or p.name)

    def finalize(self) -> "BM25Index":
        n = len(self.chunks)
        self._avgdl = (sum(self._len) / n) if n else 1.0
        self._idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }
        self._finalized = True
        return self

    # --------------------------------------------------------------- search

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Chunk]]:
        if not self._finalized:
            self.finalize()
        query_terms = words(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * self._len[i] / self._avgdl)
                score += idf * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(score, self.chunks[i]) for score, i in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.chunks)


def _import_hf_tokenizer():
    """Import Hugging Face `tokenizers.Tokenizer` safely even if cwd has a local `tokenizers/` folder."""
    import importlib
    import sys
    from pathlib import Path

    try:
        from tokenizers import Tokenizer
        return Tokenizer
    except ImportError:
        pass
    orig_path = list(sys.path)
    saved = sys.modules.pop("tokenizers", None)
    try:
        sys.path = [
            p for p in orig_path
            if ("site-packages" in p or "dist-packages" in p)
            or (p not in ("", ".") and not (Path(p) / "tokenizers").is_dir())
        ]
        mod = importlib.import_module("tokenizers")
        return mod.Tokenizer
    except Exception:
        if saved is not None:
            sys.modules["tokenizers"] = saved
        raise
    finally:
        sys.path = orig_path


class OnnxE5Embedder:
    """Lightweight CPU semantic embedder using MIT-licensed `intfloat/multilingual-e5-small` ONNX (~118 MB).

    Requires only `onnxruntime`, `tokenizers`, and `numpy` (zero transformers/sentence-transformers bloat).
    """

    MODEL_REPO = "intfloat/multilingual-e5-small"
    MODEL_FILES = (
        ("onnx/model_qint8_avx512_vnni.onnx", "model.onnx"),
        ("onnx/tokenizer.json", "tokenizer.json"),
    )

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        threads: int = 4,
        max_length: int = 256,
        allow_download: bool = True,
    ):
        import os
        import urllib.request
        from pathlib import Path

        import numpy as np
        import onnxruntime as ort

        Tokenizer = _import_hf_tokenizer()

        self.np = np
        self.max_length = max_length
        root = Path(cache_dir) if cache_dir else (Path.home() / ".cache/kuon/embedder/multilingual-e5-small")
        root.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/{self.MODEL_REPO}/resolve/main"
        for remote, local in self.MODEL_FILES:
            dst = root / local
            if not dst.is_file():
                if not allow_download:
                    raise FileNotFoundError(f"Cached embedder file missing: {dst}")
                tmp = root / (local + ".tmp")
                req = urllib.request.Request(f"{base_url}/{remote}", headers={"User-Agent": "kuon-rag/1.0"})
                with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
                    while buf := resp.read(1 << 20):
                        out.write(buf)
                os.replace(tmp, dst)

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(root / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _encode(self, texts: list[str]):
        np = self.np
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        encs = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        hidden = self.session.run(None, feeds)[0]  # (B, T, D)
        mask = attention_mask[:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)

    def embed_passages(self, texts: list[str], batch_size: int = 16):
        np = self.np
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        prefixed = [f"passage: {t}" for t in texts]
        parts = [self._encode(prefixed[i:i + batch_size]) for i in range(0, len(prefixed), batch_size)]
        return np.vstack(parts)

    def embed_query(self, query: str):
        return self._encode([f"query: {query}"])[0]


class HybridIndex:
    """Drop-in Hybrid RAG combining BM25 lexical ranking and dense semantic cosine via RRF.

    Falls back gracefully to pure BM25 when `embedder` is None.
    """

    def __init__(
        self,
        embedder=None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        min_semantic_score: float = 0.76,
    ):
        self.bm25 = BM25Index(k1=k1, b=b)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.min_semantic_score = min_semantic_score
        self._vectors = None
        self._finalized = False

    @property
    def chunks(self) -> list[Chunk]:
        return self.bm25.chunks

    def add(self, chunk: Chunk) -> None:
        self.bm25.add(chunk)
        self._finalized = False

    def add_text(self, text: str, source: str, **chunk_kwargs) -> None:
        self.bm25.add_text(text, source, **chunk_kwargs)
        self._finalized = False

    def add_file(self, path: str, source: str | None = None) -> None:
        self.bm25.add_file(path, source=source)
        self._finalized = False

    def finalize(self) -> "HybridIndex":
        self.bm25.finalize()
        if self.embedder is not None and self.bm25.chunks:
            self._vectors = self.embedder.embed_passages([c.text for c in self.bm25.chunks])
        else:
            self._vectors = None
        self._finalized = True
        return self

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Chunk]]:
        if not self._finalized:
            self.finalize()
        if not self.bm25.chunks:
            return []
        if self.embedder is None or self._vectors is None:
            return self.bm25.search(query, top_k=top_k)

        pool = max(top_k * 4, 12)
        bm25_hits = self.bm25.search(query, top_k=pool)
        chunk_to_idx = {id(c): i for i, c in enumerate(self.bm25.chunks)}
        rrf_scores: dict[int, float] = {}
        for rank, (_score, ch) in enumerate(bm25_hits, start=1):
            idx = chunk_to_idx[id(ch)]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        q_vec = self.embedder.embed_query(query)
        sims = self._vectors @ q_vec
        order = sorted(range(len(sims)), key=lambda i: float(sims[i]), reverse=True)[:pool]
        for rank, idx in enumerate(order, start=1):
            if float(sims[idx]) >= self.min_semantic_score:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.1 / (self.rrf_k + rank)

        ranked = sorted(
            rrf_scores.items(),
            key=lambda kv: (kv[1], float(sims[kv[0]])),
            reverse=True,
        )
        return [(score, self.bm25.chunks[idx]) for idx, score in ranked[:top_k]]

    def __len__(self) -> int:
        return len(self.bm25)

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
    """Split text into paragraph-aligned chunks of ~max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(Chunk(current, source))
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + "\n\n" + para).strip() if tail else para
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

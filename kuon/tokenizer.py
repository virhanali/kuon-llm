"""Byte-level BPE tokenizer, implemented from scratch.

Design (GPT-2 lineage, Llama-style byte fallback):

* Text is split into chunks with the GPT-2 regular expression
  (contractions, letter-runs, digit-runs, punctuation, whitespace).
* Chunks are UTF-8 encoded; ids 0..255 are the 256 raw bytes, so any
  unicode string can always be encoded and losslessly decoded.
* Merges are learned on a *word-frequency table*: each unique chunk is
  stored once and weighted by its frequency. This keeps training fast
  even on large corpora, and makes the learned merges independent of
  document order and repetition.
* Learned merges occupy ids 256..256+M-1. After the merges, one special
  token is appended: "<endoftext>", used as the document separator in
  packed .bin datasets.
"""

from __future__ import annotations

import codecs
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import regex

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<endoftext>"


def _merge_seq(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every adjacent `pair` in `seq` with `new_id` (left to right)."""
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(
        self,
        vocab: list[bytes],
        merges: list[tuple[int, int]],
        specials: list[str] | None = None,
    ):
        """A trained tokenizer. Use BPETokenizer.train()/load() to build one."""
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials) if specials is not None else [ENDOFTEXT]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.pattern = regex.compile(GPT2_PATTERN)
        self._encode_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------- properties

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.specials)

    @property
    def base_vocab_size(self) -> int:
        """Ids below this are byte/merge tokens; ids above are specials."""
        return len(self.vocab)

    @property
    def eos_id(self) -> int | None:
        if ENDOFTEXT in self.specials:
            return self.base_vocab_size + self.specials.index(ENDOFTEXT)
        return None

    # --------------------------------------------------------------- encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-tokenized chunk by greedily applying the
        lowest-ranked (earliest-learned) merge until none applies."""
        cached = self._encode_cache.get(chunk)
        if cached is not None:
            return cached
        seq: list[int] = list(chunk.encode("utf-8"))
        while len(seq) >= 2:
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for a, b in zip(seq, seq[1:]):
                rank = self.merge_ranks.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = (a, b), rank
            if best_pair is None:
                break
            seq = _merge_seq(seq, best_pair, 256 + best_rank)
        if len(self._encode_cache) < 1_000_000:
            self._encode_cache[chunk] = seq
        return seq

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Text -> token ids. Never emits special tokens on its own;
        pass add_eos=True when packing documents into a dataset."""
        ids: list[int] = []
        for chunk in self.pattern.findall(text):
            ids.extend(self._encode_chunk(chunk))
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text. Specials render as their literal strings."""
        parts: list[bytes] = []
        base = self.base_vocab_size
        for i in ids:
            if i >= base:
                parts.append(self.specials[i - base].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        corpus_iter,
        vocab_size: int = 8192,
        sample_limit_chars: int | None = None,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """Learn BPE merges from a text corpus (an iterable of documents).

        vocab_size counts everything: 256 byte tokens + M merges + specials.
        Training stops early if no pair occurs at least twice.
        """
        specials = [ENDOFTEXT]
        assert vocab_size >= 256 + len(specials) + 1, "vocab_size too small"

        pattern = regex.compile(GPT2_PATTERN)

        # 1) word-frequency table over a (possibly capped) sample of the corpus
        word_freq: Counter[str] = Counter()
        n_chars = 0
        n_docs = 0
        for text in corpus_iter:
            n_docs += 1
            n_chars += len(text)
            word_freq.update(pattern.findall(text))
            if sample_limit_chars is not None and n_chars >= sample_limit_chars:
                break
        if verbose:
            print(f"[tokenizer] sampled {n_docs} docs, {n_chars/1e6:.1f}M chars, "
                  f"{len(word_freq)} unique chunks")

        # 2) unique chunks -> byte sequences
        seqs: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in word_freq}

        # 3) frequency-weighted pair statistics
        pair_stats: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[str]] = defaultdict(set)
        for w, seq in seqs.items():
            f = word_freq[w]
            for a, b in zip(seq, seq[1:]):
                pair_stats[(a, b)] += f
                pair_to_words[(a, b)].add(w)

        # 4) merge loop: lazy max-heap gives O(log P) selection instead of O(P) scan
        heap: list[tuple[int, tuple[int, int]]] = [
            (-cnt, pair) for pair, cnt in pair_stats.items() if cnt > 0
        ]
        heapq.heapify(heap)
        n_merges = vocab_size - 256 - len(specials)
        merges: list[tuple[int, int]] = []
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        while len(merges) < n_merges and heap:
            neg_cnt, best = heapq.heappop(heap)
            cnt = pair_stats.get(best, 0)
            if cnt <= 0 or cnt != -neg_cnt:
                continue

            new_id = 256 + len(merges)
            merges.append(best)
            vocab.append(vocab[best[0]] + vocab[best[1]])

            touched: set[tuple[int, int]] = set()
            for w in list(pair_to_words.get(best, ())):
                seq = seqs[w]
                f = word_freq[w]
                # remove this word's old pair contributions
                for a, b in zip(seq, seq[1:]):
                    pair_stats[(a, b)] -= f
                    touched.add((a, b))
                seq = _merge_seq(seq, best, new_id)
                seqs[w] = seq
                # add the new (post-merge) pair contributions
                for a, b in zip(seq, seq[1:]):
                    pair_stats[(a, b)] += f
                    pair_to_words[(a, b)].add(w)
                    touched.add((a, b))
            pair_stats.pop(best, None)
            pair_to_words.pop(best, None)
            touched.discard(best)
            for pair in touched:
                updated = pair_stats.get(pair, 0)
                if updated > 0:
                    heapq.heappush(heap, (-updated, pair))
                else:
                    pair_stats.pop(pair, None)
                    pair_to_words.pop(pair, None)

            if verbose and len(merges) % 1000 == 0:
                print(f"[tokenizer] {len(merges)}/{n_merges} merges")

        if verbose:
            print(f"[tokenizer] learned {len(merges)} merges, "
                  f"vocab_size={256 + len(merges) + len(specials)}")
        return cls(vocab, merges, specials)

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "pattern": GPT2_PATTERN,
            "vocab": [list(b) for b in self.vocab],
            "merges": [list(m) for m in self.merges],
            "specials": self.specials,
            "vocab_size": self.vocab_size,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        raw = json.loads(Path(path).read_text())
        vocab = [bytes(v) for v in raw["vocab"]]
        merges = [tuple(m) for m in raw["merges"]]
        return cls(vocab, merges, raw.get("specials"))


class StreamDecoder:
    """Incremental UTF-8 decoder: push token ids as they are generated and
    get back printable text as soon as it is complete."""

    def __init__(self, tokenizer: BPETokenizer):
        self.tok = tokenizer
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        base = self.tok.base_vocab_size
        raw = (self.tok.specials[token_id - base].encode("utf-8")
               if token_id >= base else self.tok.vocab[token_id])
        return self.dec.decode(raw)

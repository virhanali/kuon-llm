"""Build Kuon 1's bilingual EN/ID tokenizer and clean commercial-safe corpus.

Streams only the `text` column from pinned Hugging Face Parquet shards via HTTP
Range requests (avoiding multi-GB full-shard downloads and embedding columns),
plus curated OpenStax business/economics textbooks.

Usage:
  # 1. Train balanced EN/ID 24,576-vocab BPE tokenizer (~20-40s on CPU)
  PYTHONPATH=. python scripts/prepare_kuon1_corpus.py \
    --train-tokenizer --vocab-size 24576 --out tokenizers/kuon1-bilingual-24k.json

  # 2. Pack clean bilingual bins (EN 60% incl. OpenStax Business + FineMath, ID 40% FineWeb2-HQ)
  PYTHONPATH=. python scripts/prepare_kuon1_corpus.py \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --out data/kuon1-pilot --train-tokens 20000000 --val-tokens 120000
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import time
from hashlib import blake2b, sha256
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from kuon.data import BinWriter
from kuon.tokenizer import BPETokenizer


USER_AGENT = "kuon1-corpus/1.0"

SMOLLM_REV = "3ba9d605774198c5868892d7a8deda78031a781f"
FINEWEB2_HQ_REV = "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e"
FINEMATH_REV = "e92b25a616738fe95dc186b64dfb19f9c8525594"
OPENSTAX_REV = "8f502ca45f9f05cb5673eae445b7a97a4e8c4349"
PLEIAS_SYNTH_REV = "main"

# Curated OpenStax university textbooks (CC BY 4.0) spanning both:
# 1. Core business, finance, accounting, economics, law, management, marketing, and statistics
#    (for GDPval-AA professional deliverables & KuonStudio business consulting)
# 2. Science, health/medicine, humanities, social sciences, and calculus
#    (covering the 6 domains of AA-Omniscience v1.1).
OPENSTAX_BUSINESS_FILES = (
    # Business, Finance, Accounting, Marketing, Management & Law
    "data/IntroductionToBusiness-OP_8D04gAa.txt",
    "data/Principles_Marketing-WEB.txt",
    "data/PrinciplesofManagement-OP_mGBMvoU.txt",
    "data/PrinciplesofFinance-WEB.txt",
    "data/FinancialAccounting-OP_YioY6nY.txt",
    "data/ManagerialAccounting-OP_x3x7rSO.txt",
    "data/Entrepreneurship-WEB_qNSArxT.txt",
    "data/OrganizationalBehavior-OP_TtwWIeQ.txt",
    "data/BusinessEthics-OP.txt",
    "data/Business_Law_I_Essentials_-_WEB.txt",
    "data/Introduction_to_Intellectual_Property_2PYaEdp.txt",
    "data/WritingGuide-WEB.txt",
    # Economics, Public Policy & Government
    "data/Economics3e-WEB.txt",
    "data/Microeconomics3e-WEB.txt",
    "data/Macroeconomics3e-WEB.txt",
    "data/AmericanGovernment3e-WEB.txt",
    "data/Introduction_to_Political_Science_-_WEB.txt",
    # Quantitative, Statistics & Calculus
    "data/IntroductoryBusinessStatistics-OP.txt",
    "data/IntroductoryStatistics-OP_i6tAI7e.txt",
    "data/Calculus_Volume_1_-_WEB_68M1Z5W.txt",
    "data/CalculusVolume2-OP.txt",
    # Health, Medicine & Life Sciences (AA-Omniscience Health & Science)
    "data/Anatomy_and_Physiology_2e_-_WEB_c9nD9QL.txt",
    "data/Microbiology-WEB.txt",
    "data/Biology2e-WEB_ICOFkGu.txt",
    "data/Chemistry2e-WEB.txt",
    "data/UniversityPhysicsVol1-WEB.txt",
    "data/UniversityPhysicsVolume2-WEB_5eNhMSa.txt",
    # Humanities & Social Sciences (AA-Omniscience Humanities)
    "data/Psychology2e_WEB.txt",
    "data/IntroductiontoSociology3e-WEB_9QTqRGQ.txt",
    "data/Introduction_to_Philosophy-WEB_cszrKYp.txt",
    "data/USHistory-WEB.txt",
    "data/World_History_Volume_1-WEB.txt",
    "data/World_History_Volume_2-WEB_LdwoslB.txt",
)

SOURCES_META = {
    "fineweb_edu": {
        "repo": "HuggingFaceTB/smollm-corpus",
        "subset": "fineweb-edu-dedup",
        "rev": SMOLLM_REV,
        "license": "ODC-By-1.0",
    },
    "cosmopedia_v2": {
        "repo": "HuggingFaceTB/smollm-corpus",
        "subset": "cosmopedia-v2",
        "rev": SMOLLM_REV,
        "license": "ODC-By-1.0",
    },
    "openstax_business": {
        "repo": "crumb/openstax-text",
        "files": list(OPENSTAX_BUSINESS_FILES),
        "rev": OPENSTAX_REV,
        "license": "CC-BY-4.0",
    },
    "finqa": {
        "repo": "bevaya/FinQA",
        "subset": "train",
        "rev": "main",
        "license": "MIT",
    },
    "pleias_synth": {
        "repo": "PleIAs/SYNTH",
        "subset": "en",
        "rev": PLEIAS_SYNTH_REV,
        "license": "CC-BY-4.0",
    },
    "finemath_4plus": {
        "repo": "HuggingFaceTB/finemath",
        "subset": "finemath-4plus",
        "rev": FINEMATH_REV,
        "license": "ODC-By-1.0",
    },
    "fineweb2_hq_id": {
        "repo": "epfml/FineWeb2-HQ",
        "subset": "ind_Latn",
        "rev": FINEWEB2_HQ_REV,
        "license": "ODC-By-1.0",
    },
}

# Filter out obvious boilerplate / Wikipedia footer pollution.
_BAD_MARKERS = (
    "Daftar pustaka\n",
    "Lihat pula\n",
    "Pranala luar\n",
    "Interactive Element: Thought Experiment",
)


class HttpRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader for streaming specific Parquet columns."""

    def __init__(self, url: str, timeout: int = 60, retries: int = 3):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.bytes_read = 0
        req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
            self.resolved_url = r.geturl()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n < 0 else min(self.size - 1, self.pos + n - 1)
        last_err = None
        for attempt in range(self.retries):
            try:
                req = Request(
                    self.resolved_url,
                    headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": USER_AGENT},
                )
                with urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                self.pos += len(data)
                self.bytes_read += len(data)
                return data
            except Exception as err:
                last_err = err
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"HTTP range read failed for {self.url}: {last_err}")

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def clean_text(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    if len(text) < 200:
        return None
    for marker in _BAD_MARKERS:
        if marker in text:
            return None
    return text


def iter_parquet_texts(url: str, *, batch_size: int = 256):
    handle = HttpRangeFile(url)
    pf = pq.ParquetFile(handle)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        for raw in batch.column(0).to_pylist():
            text = clean_text(raw)
            if text is not None:
                yield text


def iter_openstax_chunks(files=OPENSTAX_BUSINESS_FILES, chunk_chars: int = 3500):
    """Stream OpenStax business textbooks split into coherent multi-paragraph docs."""
    for rel_path in files:
        url = f"https://huggingface.co/datasets/crumb/openstax-text/resolve/{OPENSTAX_REV}/{rel_path}"
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=90) as r:
            raw = r.read().decode("utf-8", errors="replace")
        # Split on double newlines into clean passages ~chunk_chars long.
        paras = [p.strip() for p in re.split(r"\n{2,}", raw) if len(p.strip()) >= 80]
        buf: list[str] = []
        size = 0
        for p in paras:
            buf.append(p)
            size += len(p)
            if size >= chunk_chars:
                doc = "\n\n".join(buf)
                if len(doc) >= 200:
                    yield doc
                buf, size = [], 0
        if buf:
            doc = "\n\n".join(buf)
            if len(doc) >= 200:
                yield doc


def iter_finqa_chunks():
    """Stream bevaya/FinQA (MIT) financial report tables + reasoning steps + answers."""
    url = "https://huggingface.co/datasets/bevaya/FinQA/resolve/main/data/train-00000-of-00001.parquet"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as r:
        table = pq.read_table(io.BytesIO(r.read()))
    pre_col = table["pre_text"].to_pylist()
    tbl_col = table["table_ori"].to_pylist()
    post_col = table["post_text"].to_pylist()
    q_col = table["question"].to_pylist()
    prog_col = table["program"].to_pylist()
    ans_col = table["exe_ans"].to_pylist()
    gold_col = table["gold_inds"].to_pylist()
    for pre, tbl, post, q, prog, ans, gold in zip(
        pre_col, tbl_col, post_col, q_col, prog_col, ans_col, gold_col
    ):
        pre_s = " ".join(x.strip() for x in (pre or []) if x and x.strip() != ".")
        post_s = " ".join(x.strip() for x in (post or []) if x and x.strip() != ".")
        rows_md = []
        for r_idx, row in enumerate(tbl or []):
            cells = [str(c).strip() for c in row]
            rows_md.append("| " + " | ".join(cells) + " |")
            if r_idx == 0 and cells:
                rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
        tbl_md = "\n".join(rows_md)
        evidence = ""
        if isinstance(gold, str) and gold.strip().startswith("{"):
            try:
                g_map = json.loads(gold)
                evidence = "\n".join(f"- {v}" for v in g_map.values() if v)
            except Exception:
                pass
        doc = (
            f"Financial Report Context:\n{pre_s}\n\n"
            f"Table:\n{tbl_md}\n\n"
            f"{post_s}\n\n"
            f"Question: {q}\n"
            + (f"Relevant Facts:\n{evidence}\n" if evidence else "")
            + f"Calculation Program: {prog}\n"
            f"Verified Answer: {ans}"
        )
        cleaned = clean_text(doc)
        if cleaned is not None:
            yield cleaned


def iter_pleias_synth_texts(shard_idx: int, *, batch_size: int = 256):
    """Stream English grounded reasoning traces from PleIAs/SYNTH (CC BY 4.0)."""
    url = (
        f"https://huggingface.co/datasets/PleIAs/SYNTH/resolve/{PLEIAS_SYNTH_REV}/"
        f"synth_{shard_idx + 1:03d}.parquet"
    )
    handle = HttpRangeFile(url)
    pf = pq.ParquetFile(handle)
    cols = ["language", "query_seed_text", "query", "synthetic_reasoning", "synthetic_answer"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        langs = batch.column(0).to_pylist()
        seeds = batch.column(1).to_pylist()
        queries = batch.column(2).to_pylist()
        reasons = batch.column(3).to_pylist()
        answers = batch.column(4).to_pylist()
        for lang, seed, q, reas, ans in zip(langs, seeds, queries, reasons, answers):
            if lang != "en" or not q or not ans:
                continue
            parts = []
            if seed and str(seed).strip():
                parts.append(f"Reference Context:\n{str(seed).strip()}")
            parts.append(f"Question: {str(q).strip()}")
            if reas and str(reas).strip():
                parts.append(f"Analysis:\n{str(reas).strip()}")
            parts.append(f"Answer:\n{str(ans).strip()}")
            text = clean_text("\n\n".join(parts))
            if text is not None:
                yield text


def hf_shard_url(source_key: str, shard_idx: int) -> str:
    if source_key == "fineweb_edu":
        return (f"https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/{SMOLLM_REV}/"
                f"fineweb-edu-dedup/train-{shard_idx:05d}-of-00234.parquet")
    if source_key == "cosmopedia_v2":
        return (f"https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/{SMOLLM_REV}/"
                f"cosmopedia-v2/train-{shard_idx:05d}-of-00104.parquet")
    if source_key == "finemath_4plus":
        return (f"https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/{FINEMATH_REV}/"
                f"finemath-4plus/train-{shard_idx:05d}-of-00064.parquet")
    if source_key == "fineweb2_hq_id":
        return (f"https://huggingface.co/datasets/epfml/FineWeb2-HQ/resolve/{FINEWEB2_HQ_REV}/"
                f"ind_Latn/000_{shard_idx:05d}.parquet")
    raise ValueError(f"unknown source_key: {source_key}")


def iter_source_shards(source_key: str, shard_start: int = 0, max_shards: int = 64):
    if source_key == "openstax_business":
        yield from iter_openstax_chunks()
        return
    if source_key == "finqa":
        yield from iter_finqa_chunks()
        return
    if source_key == "pleias_synth":
        for idx in range(shard_start, shard_start + max_shards):
            yield from iter_pleias_synth_texts(idx)
        return
    for idx in range(shard_start, shard_start + max_shards):
        yield from iter_parquet_texts(hf_shard_url(source_key, idx))


def collect_chars(iterator, limit_chars: int) -> list[str]:
    docs: list[str] = []
    total = 0
    for text in iterator:
        docs.append(text)
        total += len(text)
        if total >= limit_chars:
            break
    return docs


def train_bilingual_tokenizer(out_path: Path, vocab_size: int, sample_chars: int) -> None:
    """Train a balanced EN/ID/Business/Math BPE tokenizer."""
    print(f"[kuon1] Collecting ~{sample_chars/1e6:.1f}M chars for vocab={vocab_size} tokenizer...",
          flush=True)
    # 45% ID (FineWeb2-HQ), 25% EN FineWeb-Edu, 15% OpenStax Business, 15% FineMath-4plus
    quotas = [
        ("fineweb2_hq_id", int(sample_chars * 0.45)),
        ("fineweb_edu", int(sample_chars * 0.25)),
        ("openstax_business", int(sample_chars * 0.15)),
        ("finemath_4plus", int(sample_chars * 0.15)),
    ]
    docs: list[str] = []
    for key, quota in quotas:
        part = collect_chars(iter_source_shards(key, shard_start=0), quota)
        got = sum(len(d) for d in part)
        print(f"  [sample] {key}: {len(part):,} docs ({got/1e6:.2f}M chars)", flush=True)
        docs.extend(part)

    t0 = time.time()
    tok = BPETokenizer.train(docs, vocab_size=vocab_size, verbose=True)
    tok.save(out_path)
    print(f"[kuon1] Saved tokenizer ({tok.vocab_size} vocab) to {out_path} in {time.time()-t0:.1f}s",
          flush=True)


def pack_stream(sources_with_targets: list[tuple[str, int]], val_target: int,
                out_dir: Path, lang: str, tokenizer: BPETokenizer,
                shard_offset: int) -> dict:
    """Pack multiple sources into train.{lang}.bin + val.{lang}.bin with hash-split val."""
    train_writer = BinWriter(out_dir / f"train.{lang}.bin")
    val_writer = BinWriter(out_dir / f"val.{lang}.bin")
    seen: set[bytes] = set()
    per_source: dict[str, dict[str, int]] = {}
    val_per_src_cap = max(1, val_target // max(1, len(sources_with_targets)))

    try:
        for idx_src, (src_key, train_cap) in enumerate(sources_with_targets):
            src_train = 0
            src_val = 0
            # Guarantee every source contributes to val even if earlier sources overshoot slightly.
            src_val_cap = (max(val_per_src_cap, val_target - val_writer.n_tokens)
                           if idx_src == len(sources_with_targets) - 1
                           else val_per_src_cap)
            for text in iter_source_shards(src_key, shard_start=shard_offset):
                digest = blake2b(text.encode("utf-8"), digest_size=16).digest()
                if digest in seen:
                    continue
                is_val = int.from_bytes(digest[:2], "big") % 23 == 0
                if is_val:
                    if src_val >= src_val_cap:
                        continue
                    ids = tokenizer.encode(text, add_eos=True)
                    val_writer.write(ids)
                    src_val += len(ids)
                    seen.add(digest)
                else:
                    if src_train >= train_cap:
                        if src_val >= src_val_cap:
                            break
                        continue
                    ids = tokenizer.encode(text, add_eos=True)
                    train_writer.write(ids)
                    src_train += len(ids)
                    seen.add(digest)
                if src_train >= train_cap and src_val >= src_val_cap:
                    break
            per_source[src_key] = {"train": src_train, "val": src_val}
            print(f"  [pack:{lang}] {src_key}: {src_train:,} train | {src_val:,} val", flush=True)
    finally:
        train_writer.close()
        val_writer.close()

    return {
        "train": train_writer.n_tokens,
        "val": val_writer.n_tokens,
        "by_source": per_source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-tokenizer", action="store_true",
                    help="Train a new bilingual EN/ID tokenizer and save to --out")
    ap.add_argument("--vocab-size", type=int, default=24576)
    ap.add_argument("--sample-chars", type=int, default=16_000_000,
                    help="Total characters sampled when training the tokenizer")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="Existing tokenizer.json used for corpus packing")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--val-tokens", type=int, default=120_000, help="per language")
    ap.add_argument("--shard-offset", type=int, default=0,
                    help="Starting HF shard index (increment across stages for fresh data)")
    ap.add_argument("--include-synth", action="store_true",
                    help="Include PleIAs/SYNTH (CC BY 4.0) grounded reasoning traces (~10% of EN)")
    args = ap.parse_args()

    if args.train_tokenizer:
        train_bilingual_tokenizer(args.out, args.vocab_size, args.sample_chars)
        return

    if args.tokenizer is None or not args.tokenizer.is_file():
        ap.error("--tokenizer is required when packing a corpus")
    if args.train_tokens < 10_000 or args.val_tokens < 2_000:
        ap.error("token caps are too small")

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.load(args.tokenizer)
    shutil.copyfile(args.tokenizer, args.out / "tokenizer.json")

    # Overall mix: 60% EN-side (OpenStax 33 university textbooks + FinQA financial tables +
    # optional PleIAs/SYNTH grounded reasoning + FineWeb-Edu + Cosmopedia-v2 + FineMath-4plus)
    # and 40% ID-side (FineWeb2-HQ).
    en_total = args.train_tokens * 60 // 100
    id_total = args.train_tokens - en_total
    openstax_cap = min(en_total * 10 // 60, 15_000_000)
    finqa_cap = min(en_total * 3 // 60, 2_500_000)
    synth_cap = (en_total * 10 // 100) if args.include_synth else 0
    finemath_cap = en_total * 15 // 100
    cosmo_cap = en_total * 20 // 100
    fw_edu_cap = max(1, en_total - openstax_cap - finqa_cap - synth_cap - finemath_cap - cosmo_cap)

    en_sources = [
        ("openstax_business", openstax_cap),
        ("finqa", finqa_cap),
    ]
    if synth_cap > 0:
        en_sources.append(("pleias_synth", synth_cap))
    en_sources.extend([
        ("fineweb_edu", fw_edu_cap),
        ("cosmopedia_v2", cosmo_cap),
        ("finemath_4plus", finemath_cap),
    ])
    id_sources = [("fineweb2_hq_id", id_total)]

    print(f"[kuon1] Packing {args.train_tokens:,} train tokens + {args.val_tokens:,} val/lang -> {args.out}",
          flush=True)
    en_stats = pack_stream(en_sources, args.val_tokens, args.out, "en", tokenizer, args.shard_offset)
    id_stats = pack_stream(id_sources, args.val_tokens, args.out, "id", tokenizer, args.shard_offset)

    meta = {
        "corpus": "kuon1-clean-bilingual",
        "vocab_size": tokenizer.vocab_size,
        "language_mix": {"en": 60, "id": 40},
        "shard_offset": args.shard_offset,
        "tokenizer_sha256": sha256(args.tokenizer.read_bytes()).hexdigest(),
        "tokens": {"en": en_stats, "id": id_stats},
        "sources": SOURCES_META,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[kuon1] Done: EN {en_stats['train']:,} train / {en_stats['val']:,} val | "
          f"ID {id_stats['train']:,} train / {id_stats['val']:,} val", flush=True)


if __name__ == "__main__":
    main()

"""Build task-balanced SFT data from original bilingual business scenarios."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from kuon.chat import build_example
from kuon.tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "datasets/business/scenarios.tsv"

PROMPTS = {
    "en": {
        "headings": ("Service guide", "Customer notice", "Operating notes", "Booking guide"),
        "distractors": (
            "The office updates the notice on Fridays.",
            "Questions can be sent by email.",
            "Printed copies are available at the desk.",
            "Contact details appear on the main notice.",
        ),
        "grounded": ("Cite the relevant source.", "Include the source number.",
                     "Use the excerpts and cite your answer.", "Answer briefly with the source number."),
        "abstain": ("The excerpts do not answer this question.",
                    "The available information does not say. Please check with the business.",
                    "I can’t confirm that from the documents provided.",
                    "The documents don’t specify that detail."),
        "rewrite": ("Turn this note into a short customer message:",
                    "Rewrite this for a customer chat. Do not add a promise:",
                    "Write a short announcement using only this note:",
                    "Make this update polite and concise:"),
        "opener": ("Hello. ", "Hi! ", "Please note: ", "Hello! "),
        "summary": ("Extract both facts as bullets.",
                    "Summarize the two confirmed details in one line.",
                    "List both operational details. Leave out the team chatter.",
                    "Give two short bullets for the next shift."),
        "filler": ("The team thanked everyone for their time.",
                   "The conversation ended shortly after lunch.",
                   "Someone said thanks for the update.",
                   "The team moved on to another topic."),
    },
    "id": {
        "headings": ("Panduan layanan", "Pengumuman pelanggan", "Catatan operasional", "Panduan pemesanan"),
        "distractors": (
            "Petugas memperbarui pengumuman pada hari Jumat.",
            "Pelanggan bisa mengirim pertanyaan lewat email.",
            "Salinan cetak tersedia di meja layanan.",
            "Kontak layanan tercantum dalam pengumuman utama.",
        ),
        "grounded": ("Kutip sumbernya.", "Cantumkan nomor sumber.",
                     "Jawab dari kutipan dan sebutkan sumbernya.", "Jawab singkat dengan nomor sumber."),
        "abstain": ("Kutipan itu tidak menjawab pertanyaan ini.",
                    "Informasinya belum ada. Hubungi pihak bisnis untuk memastikan.",
                    "Aku belum bisa memastikan dari dokumen yang diberikan.",
                    "Dokumen tidak menyebut rincian itu."),
        "rewrite": ("Ubah catatan ini jadi pesan singkat untuk pelanggan:",
                    "Tulis ulang untuk chat pelanggan. Jangan tambah janji:",
                    "Buat pengumuman singkat dari catatan ini saja:",
                    "Rapikan pesan ini agar sopan dan ringkas:"),
        "opener": ("Halo. ", "Halo! ", "Perlu diketahui: ", "Halo, Kak. "),
        "summary": ("Ambil kedua fakta dalam daftar berpoin.",
                    "Rangkum dua informasi yang sudah pasti dalam satu baris.",
                    "Sebutkan kedua informasi kegiatan. Abaikan obrolan tim.",
                    "Buat dua poin singkat untuk petugas sif berikutnya."),
        "filler": ("Tim mengucapkan terima kasih.",
                   "Obrolan selesai setelah makan siang.",
                   "Ada yang mengucapkan terima kasih atas kabar itu.",
                   "Tim beralih ke topik lain."),
    },
}


def examples(parts, index, language):
    question, fact, unknown, second = parts
    style, source = index % 4, 1 + (index // 2) % 2
    words = PROMPTS[language]
    main = f"{words['headings'][style]}\n{fact} {second}"
    distractor = f"{words['headings'][(style + 1) % 4]}\n{words['distractors'][style]}"
    excerpts = (main, distractor) if source == 1 else (distractor, main)
    evidence = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(excerpts, 1))
    summary = f"{fact} {second}" if style == 1 else f"- {fact}\n- {second}"

    yield {"task": "grounded", "user": f"{question} {words['grounded'][style]}",
           "assistant": f"{fact.removesuffix('.')} [{source}].", "evidence": evidence}
    yield {"task": "abstain", "user": unknown,
           "assistant": words["abstain"][(style + index // 4) % 4], "evidence": evidence}
    yield {"task": "rewrite", "user": f"{words['rewrite'][style]}\n{fact}",
           "assistant": f"{words['opener'][style]}{fact}"}
    yield {"task": "summarize",
           "user": f"{words['summary'][style]}\n{fact} {words['filler'][style]} {second}",
           "assistant": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = ROOT / "datasets/business"
    tokenizer = BPETokenizer.load(ROOT / "tokenizers/cosmopedia-16k.json")
    train, val = [], []
    split = train
    index = 0
    for line in SCENARIOS.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Development"):
            split = val
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 8:
            raise ValueError(f"Invalid scenario: {line}")
        for language, section in (("en", parts[:4]), ("id", parts[4:])):
            split.extend(dict(row, language=language, scenario=index)
                         for row in examples(section, index, language))
        index += 1

    seen = {" ".join(json.loads(line)["user"].casefold().split())
            for line in (root / "eval.jsonl").read_text(encoding="utf-8").splitlines()}
    for row in train + val:
        key = " ".join(row["user"].casefold().split())
        if key in seen:
            raise ValueError(f"Duplicate prompt: {row['user']}")
        seen.add(key)
        _, labels = build_example(tokenizer, row["user"], row["assistant"], 512,
                                  evidence=row.get("evidence"))
        if not labels or labels[-1] != tokenizer.eos_id:
            raise ValueError(f"Example does not fit: {row['user']}")

    rng = random.Random(1337)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        rng.shuffle(rows)
        with (args.out / f"sft_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, len(rows), dict(sorted(Counter((r["language"], r["task"]) for r in rows).items())))


if __name__ == "__main__":
    main()

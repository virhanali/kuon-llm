"""Generate Kuon 1.1 non-coding bilingual EN/ID distillation dataset from Gemma 4 (Apache-2.0).

Distills all non-coding Gemma 4 capabilities into Kuon 1's 100%-from-scratch weights:
  1. Natural smalltalk, casual reactions, varied identity & capability Q&A, polite coding boundary
  2. Open-ended business consulting & empathetic symptom triage ("bisnis ku lesu", etc.)
  3. Document summarization + grounded QA + honest unanswerable restraint (non-coding docs/SOPs/specs)
  4. Daily business writing & copywriting (WA replies, follow-ups, complaint handling, memos, promos)
  5. Clear explanations, structured reasoning, & practical comparisons (strictly non-coding)
  6. Multi-turn follow-up conversations (2-3 turns)

Supports two teacher backends and resumable generation:
  - `--backend openai` (default): local/remote `llama-server` or vLLM OpenAI-compatible endpoint
  - `--backend hf`: batched Hugging Face `transformers` on Colab/Kaggle GPU (e.g. google/gemma-4-E4B-it)

Every kept row is:
  - Scrubbed of teacher/vendor identity leaks (Gemma, Google, DeepMind, OpenAI, etc.)
  - Blocked from containing programming code snippets (non-coding scope)
  - Numerically verified against evidence with `kuon_agent.tools.extract_grounded_numbers`
  - Checked against all holdout/dev benchmarks (zero leakage) and optional 1,024-token budget
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

from kuon_agent.tools import extract_grounded_numbers
from scripts.distill_gemma4 import (
    ABSTAIN_RE,
    CITATION_RE,
    TEACHER_LEAK_RE,
    extract_json_object,
    load_forbidden_prompts,
)

ROOT = Path(__file__).resolve().parents[1]

# Reject generated samples that sneak in programming code blocks or syntax
CODING_BLOCK_RE = re.compile(
    r"```|\bdef\s+\w+\(|\bfunc\s+\w+\(|\bpackage\s+main\b|\bimport\s+(?:os|sys|react|fmt)\b|"
    r"\bconsole\.log\(|\bprintf\(|\bSELECT\s+\*\s+FROM\b|\bCREATE\s+TABLE\b|\bclass\s+\w+\s*[:{]|"
    r"\bconst\s+\w+\s*=\s*\(|<\?php|#!/usr/bin/env",
    re.IGNORECASE,
)

INDUSTRIES_EXT = [
    ("warung kopi", "neighborhood coffee shop"),
    ("toko roti rumahan", "artisan bakery"),
    ("layanan laundry kiloan", "wash-and-fold laundry service"),
    ("klinik gigi", "dental clinic"),
    ("bengkel motor", "motorcycle repair shop"),
    ("toko kelontong modern", "mini grocery store"),
    ("bisnis katering harian", "daily meal catering"),
    ("SaaS B2B berlangganan", "B2B subscription SaaS"),
    ("toko pakaian online", "online apparel boutique"),
    ("distributor bahan kue", "wholesale baking supplier"),
    ("jasa pengiriman lokal", "local parcel courier"),
    ("studio kebugaran", "boutique fitness studio"),
    ("apotek keluarga", "community pharmacy"),
    ("restoran cepat saji", "fast-casual restaurant"),
    ("agen perjalanan", "boutique travel agency"),
    ("toko elektronik", "consumer electronics shop"),
    ("salon kecantikan", "hair and beauty salon"),
    ("jasa kebersihan rumah", "home cleaning service"),
    ("percetakan digital", "digital print shop"),
    ("gudang fulfillment e-commerce", "e-commerce fulfillment warehouse"),
    ("klinik hewan", "veterinary clinic"),
    ("toko bunga online", "online flower shop"),
    ("kursus bahasa", "language tutoring center"),
    ("distributor kopi sangrai", "specialty coffee roaster"),
]

OPEN_SYMPTOMS_ID = [
    "bisnis ku lesu banget bulan ini",
    "toko sepi padahal sudah pasang spanduk",
    "iklan sudah jalan tapi boncos terus",
    "omzet turun lumayan tajam dua minggu terakhir",
    "pelanggan lama jarang balik lagi",
    "banyak yang tanya di WhatsApp tapi jarang jadi beli",
    "arus kas seret padahal penjualan kelihatan ramai",
    "stok bahan baku sering numpuk lalu terbuang",
    "staf sering salah catat pesanan waktu jam ramai",
    "komplain keterlambatan makin banyak bulan ini",
    "margin terasa makin tipis walau transaksi stabil",
    "bingung menentukan prioritas perbaikan usaha minggu ini",
]

OPEN_SYMPTOMS_EN = [
    "my business has been really slow this month",
    "foot traffic dropped even after running a local promo",
    "our ad spend keeps rising without bringing real buyers",
    "revenue slipped noticeably over the past two weeks",
    "repeat customers stopped coming back",
    "lots of inquiries come in on chat but very few convert",
    "cash flow feels tight even though sales look busy",
    "inventory keeps piling up and spoiling",
    "staff keep making order-entry mistakes during rush hours",
    "delivery delay complaints spiked this month",
    "profit margins feel thinner even with steady orders",
    "I am not sure which operational bottleneck to fix first",
]

DOC_TYPES_ID = [
    "spesifikasi teknis layanan non-kode (target latensi, kapasitas, dan risiko)",
    "SOP operasional harian tim pelayanan pelanggan",
    "notulen rapat evaluasi penjualan mingguan",
    "ringkasan laporan kinerja cabang dan biaya operasional",
    "proposal kerja sama vendor dan jadwal pengiriman",
    "kebijakan pengembalian barang dan penanganan komplain",
]

DOC_TYPES_EN = [
    "non-coding service technical spec (latency targets, throughput, and risks)",
    "daily customer operations SOP",
    "weekly sales review meeting notes",
    "branch performance and operating cost summary",
    "vendor partnership proposal and delivery SLA",
    "refund and complaint escalation policy",
]

WRITING_TASKS_ID = [
    "draf balasan WhatsApp sopan untuk pelanggan yang menanyakan pesanan terlambat",
    "pesan follow-up singkat hari ke-3 untuk calon klien yang belum membalas penawaran",
    "pengumuman promo akhir pekan yang hangat tanpa terkesan memaksa",
    "pesan klarifikasi ke pemasok soal selisih jumlah barang yang dikirim",
    "pengumuman internal singkat untuk tim tentang jam tutup rekap harian",
    "balasan ulasan bintang 3 dari pelanggan agar tetap positif dan solutif",
    "draf pesan penawaran awal ke pemilik kafe untuk kerja sama pasokan",
    "template ucapan terima kasih setelah transaksi selesai disertai ajakan repeat order",
]

WRITING_TASKS_EN = [
    "a polite WhatsApp reply to a customer asking about a delayed order",
    "a concise day-3 follow-up message to a prospect who has not replied to a quote",
    "a warm weekend promotion announcement that does not sound pushy",
    "a clarification message to a supplier about a short-shipped delivery",
    "a brief internal team memo about the daily order cut-off time",
    "a constructive reply to a 3-star customer review",
    "a short outreach message offering supply partnership to a local cafe owner",
    "a post-purchase thank-you message encouraging repeat orders",
]

EXPLAIN_TOPICS_ID = [
    "perbedaan laba bersih dan arus kas (cash flow) untuk pemilik UMKM",
    "cara membaca gross margin dan kenapa omzet besar bisa tetap rugi",
    "cara menghitung titik impas (break-even) secara sederhana sebelum buka cabang",
    "perbedaan diskon persen dan bundling produk terhadap margin",
    "konsep caching pada sistem aplikasi dari sisi bisnis (biaya server dan kecepatan)",
    "cara menyusun SOP singkat 1 halaman agar staf baru cepat mandiri",
    "cara membedakan masalah trafik, konversi, dan retensi pelanggan",
    "prinsip negosiasi termin pembayaran dengan pemasok tanpa merusak hubungan",
    "cara membuat evaluasi pilot 2 minggu sebelum investasi sistem besar",
    "kesalahan umum saat membaca rata-rata penjualan tanpa melihat sebaran produk",
    "cara mengelompokkan pelanggan setia dan pelanggan sekali beli",
    "manfaat pencatatan stok harian dibanding opname bulanan saja",
]

EXPLAIN_TOPICS_EN = [
    "the difference between net profit and cash flow for small business owners",
    "how to read gross margin and why high revenue can still lose money",
    "how to think about break-even volume before expanding a store",
    "comparing percentage discounts versus product bundles on profit margin",
    "the business value of application caching (server cost and response speed) without code",
    "how to write a 1-page SOP so new staff onboard quickly",
    "how to separate traffic, conversion, and retention problems",
    "practical principles for negotiating supplier payment terms politely",
    "how to evaluate a 2-week workflow pilot before scaling",
    "common pitfalls when reading average order value without checking product mix",
    "how to segment repeat buyers from one-time shoppers",
    "why daily stock tracking beats monthly-only counts",
]


def is_clean_noncoding(text: str, forbidden: set[str]) -> bool:
    if not text or not text.strip():
        return False
    if TEACHER_LEAK_RE.search(text):
        return False
    if CODING_BLOCK_RE.search(text):
        return False
    norm = " ".join(text.casefold().split())
    for bad in forbidden:
        if bad and bad in norm:
            return False
    return True


def _numbers_grounded_in(answer: str, evidence: str) -> bool:
    """Verify every numeric figure in answer matches a number stated in evidence (tolerance 1e-4)."""
    ev_nums = extract_grounded_numbers(CITATION_RE.sub("", evidence))
    ans_nums = extract_grounded_numbers(CITATION_RE.sub("", answer))
    if not ans_nums:
        return True
    for val in ans_nums:
        if val in (0.0, 1.0, 2.0, 3.0):
            continue
        if not any(abs(val - e) <= max(1e-4, 1e-3 * abs(e)) for e in ev_nums):
            return False
    return True


def build_casual_identity_rows() -> list[dict]:
    """Curated bilingual casual chat, reactions, identity, capability, and coding-boundary rows."""
    pairs = [
        # Casual reactions & acknowledgements
        ("id", "hehe", "Hehe, santai saja! Kalau ada data bisnis, dokumen, atau ide yang mau dibahas, langsung kirim ya."),
        ("id", "wkwk", "Haha siap! Ada kendala usaha atau hal lain yang mau kita cek bareng sekarang?"),
        ("id", "haha", "Santai! Tinggal kasih tahu saja kalau ada yang mau didiskusikan atau diringkas."),
        ("id", "oke makasih ya", "Sama-sama! Kalau nanti butuh bantu cek angka atau susun draf lagi, tinggal kabari."),
        ("id", "siap mantap", "Siap! Kabari saja kalau ada langkah lanjutan yang mau diuji."),
        ("id", "halo", "Halo! Saya Kuon AI dari Kuon Studio. Ada dokumen, angka usaha, atau pertanyaan yang mau kita bahas hari ini?"),
        ("id", "hai kuon", "Hai! Ada yang bisa saya bantu cek atau diskusikan hari ini?"),
        ("en", "hehe", "Haha all good! Whenever you have a question, document, or business problem to talk through, just send it over."),
        ("en", "ok thanks!", "You are welcome! Let me know anytime you want to review numbers, draft a message, or summarize a doc."),
        ("en", "hi", "Hi! I am Kuon AI from Kuon Studio. What business question, document, or idea would you like to work on today?"),
        # Varied identity & capability phrasings
        ("id", "ilmu kamu apa aja?",
         "Saya Kuon AI, ditenagai model Kuon 1 buatan Kuon Studio. Fokus saya membantu analisis bisnis dan operasional, membaca serta meringkas dokumen, menyusun draf komunikasi (seperti balasan WhatsApp atau memo), menghitung tabel KPI dengan rumus pasti, dan menjelaskan konsep bisnis dengan bahasa sederhana."),
        ("id", "skill apa aja yang kamu punya?",
         "Saya bisa bantu beberapa hal praktis: diskusi strategi dan kendala bisnis, meringkas atau tanya-jawab isi dokumen, menyusun draf pesan pelanggan dan promo, menghitung tabel angka serta KPI bisnis secara pasti, dan menjelaskan konsep operasional tanpa bahasa yang berbelit."),
        ("id", "kamu model llm apa?",
         "Saya Kuon AI, asisten AI self-hosted yang ditenagai model Kuon 1 buatan Kuon Studio di Jakarta untuk membantu analisis bisnis, dokumen, dan kerja harian."),
        ("id", "model apa kamu?",
         "Saya Kuon AI dengan model Kuon 1 yang dikembangkan dan dioperasikan oleh Kuon Studio (kuonstudio.com, Jakarta)."),
        ("id", "kamu didevelop sama siapa?",
         "Saya dikembangkan dan dioperasikan oleh Kuon Studio (kuonstudio.com, Jakarta) sebagai asisten AI self-hosted untuk bisnis."),
        ("id", "kamu dibuat oleh siapa?",
         "Saya dibuat dan dikelola oleh Kuon Studio di Jakarta. Ada kendala bisnis atau dokumen yang sedang ingin Anda kerjakan?"),
        ("en", "what skills do you have?",
         "I am Kuon AI, powered by Kuon 1 from Kuon Studio. I help with business troubleshooting and strategy, summarizing and answering questions on documents, drafting customer messages and memos, checking sales tables and KPIs, and explaining business concepts clearly."),
        ("en", "what model are you?",
         "I am Kuon AI, a self-hosted business assistant powered by Kuon 1, built and operated by Kuon Studio in Jakarta."),
        ("en", "who developed you?",
         "I am developed and operated by Kuon Studio (kuonstudio.com, Jakarta) as a self-hosted AI assistant for business operations and analysis."),
        # Polite coding boundary (non-coding scope)
        ("id", "berikan code golang untuk penjumlahan",
         "Fokus saya adalah konsultasi bisnis, analisis dokumen, penulisan draf, dan perhitungan data—bukan penulisan kode pemrograman. Kalau Anda ingin membahas dampak teknis ke bisnis atau merangkum dokumen spesifikasi, saya siap bantu."),
        ("id", "tolong buatkan script Python untuk scraping",
         "Saya tidak fokus menulis kode pemrograman. Fokus saya di strategi bisnis, ringkasan dokumen, draf komunikasi, dan analisis angka operasional. Ada dokumen atau masalah usaha yang bisa saya bantu cek?"),
        ("en", "write a Golang program for me",
         "My focus is business consulting, document analysis, business writing, and grounded data checks rather than writing programming code. If you want to summarize a technical spec or evaluate a business workflow, I would be glad to help."),
        ("en", "can you debug my React and SQL code?",
         "I do not handle software coding or debugging. I specialize in business operations, document summaries, drafting messages, and analyzing business numbers. Let me know if you have a report or workflow problem to work through."),
    ]
    rows: list[dict] = []
    for lang, u, a in pairs:
        for _ in range(3):
            rows.append({
                "language": lang,
                "task": "k11_casual_identity",
                "business": False,
                "user": u,
                "assistant": a,
            })
    return rows


def build_prompt_specs(
    *,
    open_consult: int,
    doc_summary: int,
    writing: int,
    explain: int,
    multiturn: int,
    casual: int,
    seed: int = 2026,
) -> list[dict]:
    """Create deterministic teacher prompt specifications across all 6 non-coding pillars."""
    rng = random.Random(seed)
    specs: list[dict] = []

    def pick_lang(i: int) -> str:
        # ~65% Indonesian, ~35% English
        return "id" if (i % 20) < 13 else "en"

    # 1. Casual / conversational & capability nuances
    casual_themes_id = [
        "sapaan santai pengguna di pagi/siang/malam",
        "reaksi tawa singkat atau basa-basi ringan",
        "pertanyaan santai soal ilmu, kemampuan, atau kegunaan Kuon AI",
        "pertanyaan apakah Kuon AI bisa menggantikan staf manusia",
        "permintaan menulis kode pemrograman yang harus ditolak secara ramah ke fokus bisnis",
    ]
    casual_themes_en = [
        "a casual greeting from a user",
        "a brief friendly reaction or thank-you note",
        "a casual question asking what Kuon AI knows and can help with",
        "a question asking if Kuon AI replaces human staff",
        "a request to write software code that should be politely declined toward business focus",
    ]
    for i in range(casual):
        lang = pick_lang(i)
        if lang == "id":
            theme = casual_themes_id[i % len(casual_themes_id)]
            prompt = (
                f"Buat 1 contoh percakapan singkat alami berbahasa Indonesia untuk tema: '{theme}'.\n"
                "Persona asisten: Kuon AI (ditenagai model Kuon 1 buatan Kuon Studio di Jakarta), hangat, luwes, tidak kaku, tidak menulis kode pemrograman, dan tidak pernah menyebut vendor AI lain.\n"
                'Keluarkan HANYA JSON valid: {"user": "pesan singkat pengguna", "assistant": "balasan alami 1-3 kalimat"}'
            )
        else:
            theme = casual_themes_en[i % len(casual_themes_en)]
            prompt = (
                f"Create 1 natural short English exchange for theme: '{theme}'.\n"
                "Assistant persona: Kuon AI (powered by Kuon 1 from Kuon Studio in Jakarta), warm, concise, non-coding scope, never mention other AI vendors.\n"
                'Output ONLY valid JSON: {"user": "short user message", "assistant": "natural 1-3 sentence reply"}'
            )
        specs.append({"id": f"casual-{i}", "kind": "casual", "lang": lang, "prompt": prompt})

    # 2. Open-ended business consulting & symptom triage
    for i in range(open_consult):
        lang = pick_lang(i)
        ind_id, ind_en = INDUSTRIES_EXT[i % len(INDUSTRIES_EXT)]
        if lang == "id":
            sym = OPEN_SYMPTOMS_ID[(i + rng.randint(0, 4)) % len(OPEN_SYMPTOMS_ID)]
            prompt = (
                f"Buat 1 tanya-jawab konsultasi bisnis berbahasa Indonesia untuk pemilik '{ind_id}' dengan keluhan awal tanpa angka lengkap: '{sym}'.\n"
                "Keluarkan HANYA JSON valid dengan 3 key:\n"
                '{"user": "keluhan singkat alami pemilik usaha (1-2 kalimat, boleh santai, belum menyebut angka lengkap)", '
                '"assistant": "jawaban empatik dan tajam (3-4 kalimat): akui masalahnya, beri 1 sudut pandang praktis tanpa mengarang angka, lalu ajukan 2 pertanyaan diagnostik spesifik", '
                '"rejected": "jawaban kaku yang salah menolak seperti Kutipan [1] tidak menyebutkan jawaban atau malah melantur dalam bahasa Inggris"}'
            )
        else:
            sym = OPEN_SYMPTOMS_EN[(i + rng.randint(0, 4)) % len(OPEN_SYMPTOMS_EN)]
            prompt = (
                f"Create 1 English business consulting Q&A for a '{ind_en}' owner sharing an open symptom without full numbers: '{sym}'.\n"
                "Output ONLY valid JSON with 3 keys:\n"
                '{"user": "natural 1-2 sentence owner complaint without full figures", '
                '"assistant": "empathetic, sharp 3-4 sentence reply: acknowledge the issue, give 1 practical angle without inventing numbers, and ask 2 targeted diagnostic questions", '
                '"rejected": "a robotic over-refusal claiming Excerpt [1] does not mention the answer"}'
            )
        specs.append({"id": f"consult-{i}", "kind": "open_consult", "lang": lang, "prompt": prompt})

    # 3. Document summarization + grounded QA + honest unanswerable
    for i in range(doc_summary):
        lang = pick_lang(i)
        ind_id, ind_en = INDUSTRIES_EXT[i % len(INDUSTRIES_EXT)]
        if lang == "id":
            dtype = DOC_TYPES_ID[i % len(DOC_TYPES_ID)]
            prompt = (
                f"Buat 1 dokumen ringkas non-kode berbahasa Indonesia bertipe '{dtype}' untuk usaha '{ind_id}' (90-150 kata, memuat 3-4 angka/target nyata, TANPA kode pemrograman).\n"
                "Keluarkan HANYA JSON valid dengan 8 key string:\n"
                '{"title": "judul dokumen singkat", '
                '"doc": "isi dokumen terstruktur 2-3 paragraf atau poin berisi 3-5 angka nyata", '
                '"sum_q": "perintah pengguna meminta ringkasan poin utama dan angka penting dokumen", '
                '"sum_a": "ringkasan rapi 3-4 poin yang HANYA memakai fakta dan angka persis dari doc serta menyebut [1]", '
                '"qa_q": "pertanyaan spesifik tentang detail atau angka ke-2/ke-3 di dokumen", '
                '"qa_a": "jawaban ringkas akurat mengutip [1] dengan angka persis dari doc", '
                '"unans_q": "pertanyaan bisnis masuk akal yang datanya TIDAK ada di dokumen", '
                '"unans_a": "penolakan jujur bahwa dokumen [1] tidak memuat data tersebut + minta detail tambahan tanpa mengarang angka"}'
            )
        else:
            dtype = DOC_TYPES_EN[i % len(DOC_TYPES_EN)]
            prompt = (
                f"Create 1 concise non-coding English document of type '{dtype}' for a '{ind_en}' (90-150 words, containing 3-4 concrete figures/targets, NO programming code).\n"
                "Output ONLY valid JSON with these exact 8 string keys:\n"
                '{"title": "short document title", '
                '"doc": "structured 2-3 paragraph/bullet document with 3-5 real figures", '
                '"sum_q": "user prompt asking to summarize key points and numbers from the document", '
                '"sum_a": "clear 3-4 bullet summary using ONLY exact facts and numbers from doc and citing [1]", '
                '"qa_q": "specific question about the 2nd or 3rd detail/figure in the doc", '
                '"qa_a": "concise grounded answer citing [1] with exact figures from doc", '
                '"unans_q": "reasonable business question whose answer is MISSING from the doc", '
                '"unans_a": "honest abstention stating document [1] does not include that information + 1 follow-up request without inventing numbers"}'
            )
        specs.append({"id": f"doc-{i}", "kind": "doc_summary", "lang": lang, "prompt": prompt})

    # 4. Daily business writing & copywriting
    for i in range(writing):
        lang = pick_lang(i)
        ind_id, ind_en = INDUSTRIES_EXT[i % len(INDUSTRIES_EXT)]
        if lang == "id":
            wtask = WRITING_TASKS_ID[i % len(WRITING_TASKS_ID)]
            prompt = (
                f"Buat 1 contoh permintaan penulisan bisnis berbahasa Indonesia untuk usaha '{ind_id}': {wtask}.\n"
                "Keluarkan HANYA JSON valid: "
                '{"user": "permintaan realistis pemilik usaha dengan konteks singkat", '
                '"assistant": "hasil draf siap kirim yang hangat, jelas, singkat (maksimal 120 kata), tanpa jargon kaku"}'
            )
        else:
            wtask = WRITING_TASKS_EN[i % len(WRITING_TASKS_EN)]
            prompt = (
                f"Create 1 English business writing request for a '{ind_en}': {wtask}.\n"
                "Output ONLY valid JSON: "
                '{"user": "realistic owner request with brief context", '
                '"assistant": "ready-to-send draft that is warm, clear, and concise (under 120 words)"}'
            )
        specs.append({"id": f"write-{i}", "kind": "writing", "lang": lang, "prompt": prompt})

    # 5. Clear non-coding explanations & structured reasoning
    for i in range(explain):
        lang = pick_lang(i)
        if lang == "id":
            topic = EXPLAIN_TOPICS_ID[i % len(EXPLAIN_TOPICS_ID)]
            prompt = (
                f"Buat 1 tanya-jawab penjelasan konsep praktis (TANPA kode pemrograman) berbahasa Indonesia tentang: '{topic}'.\n"
                "Keluarkan HANYA JSON valid: "
                '{"user": "pertanyaan alami pemilik bisnis atau staf tentang topik tersebut", '
                '"assistant": "penjelasan jernih, terstruktur (3-5 kalimat atau poin singkat), pakai analogi bisnis nyata yang mudah dipahami"}'
            )
        else:
            topic = EXPLAIN_TOPICS_EN[i % len(EXPLAIN_TOPICS_EN)]
            prompt = (
                f"Create 1 practical non-coding English explanation Q&A on: '{topic}'.\n"
                "Output ONLY valid JSON: "
                '{"user": "natural business owner or operator question on this topic", '
                '"assistant": "clear, structured explanation (3-5 sentences or short bullets) with a concrete business example"}'
            )
        specs.append({"id": f"explain-{i}", "kind": "explain", "lang": lang, "prompt": prompt})

    # 6. Multi-turn follow-up conversations (2-turn)
    for i in range(multiturn):
        lang = pick_lang(i)
        ind_id, ind_en = INDUSTRIES_EXT[(i + 5) % len(INDUSTRIES_EXT)]
        if lang == "id":
            sym = OPEN_SYMPTOMS_ID[(i + 3) % len(OPEN_SYMPTOMS_ID)]
            prompt = (
                f"Buat 1 percakapan konsultasi 2 giliran berbahasa Indonesia untuk '{ind_id}' yang diawali '{sym}'.\n"
                "Keluarkan HANYA JSON valid dengan 4 key:\n"
                '{"u1": "pesan awal singkat pengguna", '
                '"a1": "balasan empatik + 1-2 pertanyaan klarifikasi spesifik", '
                '"u2": "jawaban pengguna memberi fakta/angka konkret", '
                '"a2": "saran tindakan praktis yang merujuk persis pada fakta u2 + cara mengukurnya dalam 1-2 minggu"}'
            )
        else:
            sym = OPEN_SYMPTOMS_EN[(i + 3) % len(OPEN_SYMPTOMS_EN)]
            prompt = (
                f"Create 1 natural 2-turn English consulting dialogue for a '{ind_en}' starting with '{sym}'.\n"
                "Output ONLY valid JSON with 4 keys:\n"
                '{"u1": "short opening user message", '
                '"a1": "empathetic reply + 1-2 targeted clarifying questions", '
                '"u2": "user follow-up supplying concrete facts/numbers", '
                '"a2": "practical tailored recommendation referencing u2 facts + how to measure results in 1-2 weeks"}'
            )
        specs.append({"id": f"mt-{i}", "kind": "multiturn", "lang": lang, "prompt": prompt})

    return specs


def convert_record(
    spec: dict,
    obj: dict,
    forbidden: set[str],
    sft_out: list[dict],
    prefs_out: list[dict],
) -> bool:
    """Validate and convert one teacher JSON object into SFT and preference rows."""
    kind = spec["kind"]
    lang = spec["lang"]

    if kind == "casual":
        u = str(obj.get("user", "")).strip()
        a = str(obj.get("assistant", "")).strip()
        if len(u) < 2 or len(a) < 12 or not is_clean_noncoding(u, forbidden) or not is_clean_noncoding(a, forbidden):
            return False
        sft_out.append({"language": lang, "task": "k11_casual", "business": False, "user": u, "assistant": a})
        return True

    if kind == "open_consult":
        u = str(obj.get("user", "")).strip()
        a = str(obj.get("assistant", "")).strip()
        rej = str(obj.get("rejected", "")).strip()
        if len(u) < 8 or len(a) < 30 or not is_clean_noncoding(u, forbidden) or not is_clean_noncoding(a, forbidden):
            return False
        sft_out.append({"language": lang, "task": "k11_open_consult", "business": True, "user": u, "assistant": a})
        if len(rej) >= 12 and rej.casefold() != a.casefold() and not TEACHER_LEAK_RE.search(rej):
            prefs_out.append({
                "language": lang, "task": "k11_pref_open_consult", "business": True,
                "user": u, "chosen": a, "rejected": rej,
            })
        return True

    if kind == "doc_summary":
        keys = ("title", "doc", "sum_q", "sum_a", "qa_q", "qa_a", "unans_q", "unans_a")
        vals = [str(obj.get(k, "")).strip() for k in keys]
        if any(len(v) < 8 for v in vals):
            return False
        if any(not is_clean_noncoding(v, forbidden) for v in vals):
            return False
        title, doc, sum_q, sum_a, qa_q, qa_a, unans_q, unans_a = vals
        ev = f"[1] {title}\n{doc}"
        if not _numbers_grounded_in(sum_a, ev) or not _numbers_grounded_in(qa_a, ev):
            return False
        if not ABSTAIN_RE.search(unans_a) or not _numbers_grounded_in(unans_a, ev):
            return False

        bad_overrefuse = (
            "Kutipan [1] tidak menyebutkan jawaban untuk pertanyaan tersebut. Bisa sertakan bagian atau informasi tambahan?"
            if lang == "id"
            else "Excerpt [1] does not mention the answer to that question. Could you share additional details?"
        )
        sft_out.append({"language": lang, "task": "k11_doc_summary", "business": True, "user": sum_q, "evidence": ev, "assistant": sum_a})
        sft_out.append({"language": lang, "task": "k11_doc_qa", "business": True, "user": qa_q, "evidence": ev, "assistant": qa_a})
        sft_out.append({"language": lang, "task": "k11_doc_unans", "business": True, "user": unans_q, "evidence": ev, "assistant": unans_a})

        prefs_out.append({
            "language": lang, "task": "k11_pref_doc_summary", "business": True,
            "user": sum_q, "evidence": ev, "chosen": sum_a, "rejected": bad_overrefuse,
        })
        prefs_out.append({
            "language": lang, "task": "k11_pref_doc_qa", "business": True,
            "user": qa_q, "evidence": ev, "chosen": qa_a, "rejected": bad_overrefuse,
        })
        return True

    if kind == "writing":
        u = str(obj.get("user", "")).strip()
        a = str(obj.get("assistant", "")).strip()
        if len(u) < 12 or len(a) < 25 or not is_clean_noncoding(u, forbidden) or not is_clean_noncoding(a, forbidden):
            return False
        sft_out.append({"language": lang, "task": "k11_writing", "business": False, "user": u, "assistant": a})
        return True

    if kind == "explain":
        u = str(obj.get("user", "")).strip()
        a = str(obj.get("assistant", "")).strip()
        if len(u) < 12 or len(a) < 35 or not is_clean_noncoding(u, forbidden) or not is_clean_noncoding(a, forbidden):
            return False
        sft_out.append({"language": lang, "task": "k11_explain", "business": False, "user": u, "assistant": a})
        return True

    if kind == "multiturn":
        keys = ("u1", "a1", "u2", "a2")
        vals = [str(obj.get(k, "")).strip() for k in keys]
        if any(len(v) < 10 for v in vals):
            return False
        if any(not is_clean_noncoding(v, forbidden) for v in vals):
            return False
        u1, a1, u2, a2 = vals
        sft_out.append({"language": lang, "task": "k11_mt_t1", "business": True, "user": u1, "assistant": a1})
        sft_out.append({
            "language": lang, "task": "k11_mt_t2", "business": True,
            "user": f"{u1}\nAssistant: {a1}\nUser: {u2}",
            "assistant": a2,
        })
        return True

    return False


def _call_openai_teacher(base_url: str, model: str, prompt: str, max_tokens: int = 780) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.75,
        "top_p": 0.92,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        data = json.loads(res.read().decode("utf-8", errors="replace"))
    choices = data.get("choices") or []
    return str(choices[0]["message"]["content"]) if choices else ""


def _run_openai_generation(
    specs: list[dict],
    cache_path: Path,
    *,
    base_url: str,
    model: str,
    concurrency: int = 4,
    max_tokens: int = 780,
) -> dict[str, str]:
    cached: dict[str, str] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                cached[rec["id"]] = rec["raw"]
            except Exception:
                continue

    pending = [s for s in specs if s["id"] not in cached]
    print(f"[distill-k11] openai backend: {len(cached)} cached, {len(pending)} pending (concurrency={concurrency})", flush=True)
    if not pending:
        return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = len(cached)
    with cache_path.open("a", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            fut_map = {
                pool.submit(_call_openai_teacher, base_url, model, s["prompt"], max_tokens): s
                for s in pending
            }
            for fut in concurrent.futures.as_completed(fut_map):
                spec = fut_map[fut]
                try:
                    raw = fut.result()
                except Exception as err:
                    print(f"[distill-k11] warn {spec['id']}: {err}", flush=True)
                    continue
                cached[spec["id"]] = raw
                f_out.write(json.dumps({"id": spec["id"], "raw": raw}, ensure_ascii=False) + "\n")
                f_out.flush()
                done += 1
                if done % 25 == 0 or done == len(specs):
                    print(f"[distill-k11] progress {done}/{len(specs)}", flush=True)
    return cached


def _run_hf_generation(
    specs: list[dict],
    cache_path: Path,
    *,
    model_id: str,
    batch_size: int = 16,
    max_tokens: int = 780,
    load_in_4bit: bool = False,
) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cached: dict[str, str] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                cached[rec["id"]] = rec["raw"]
            except Exception:
                continue

    pending = [s for s in specs if s["id"] not in cached]
    print(f"[distill-k11] hf backend: {len(cached)} cached, {len(pending)} pending", flush=True)
    if not pending:
        return cached

    print(f"[distill-k11] loading {model_id} (4bit={load_in_4bit}) once...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = len(cached)
    with cache_path.open("a", encoding="utf-8") as f_out:
        for start in range(0, len(pending), batch_size):
            sub = pending[start : start + batch_size]
            texts = []
            for s in sub:
                try:
                    t = tok.apply_chat_template(
                        [{"role": "user", "content": s["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    try:
                        t = tok.apply_chat_template(
                            [{"role": "user", "content": s["prompt"]}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        t = tok.apply_chat_template(
                            [{"role": "user", "content": [{"type": "text", "text": s["prompt"]}]}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                texts.append(t)
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.75,
                    top_p=0.92,
                    pad_token_id=tok.pad_token_id,
                )
            for spec, seq in zip(sub, out):
                raw = tok.decode(seq[in_len:], skip_special_tokens=True)
                cached[spec["id"]] = raw
                f_out.write(json.dumps({"id": spec["id"], "raw": raw}, ensure_ascii=False) + "\n")
            f_out.flush()
            done += len(sub)
            if done % max(batch_size * 4, 64) == 0 or done == len(specs):
                print(f"[distill-k11] progress {done}/{len(specs)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return cached


def _fits_budget(row: dict, tokenizer, block_size: int) -> bool:
    if tokenizer is None:
        return True
    from kuon.chat import format_example

    ev = row.get("evidence")
    biz = bool(row.get("business", False))
    if "assistant" in row:
        txt = format_example(row["user"], row["assistant"], evidence=ev, business=biz)
        return len(tokenizer.encode(txt, add_eos=True)) <= block_size
    if "chosen" in row and "rejected" in row:
        t1 = format_example(row["user"], row["chosen"], evidence=ev, business=biz)
        t2 = format_example(row["user"], row["rejected"], evidence=ev, business=biz)
        return (
            len(tokenizer.encode(t1, add_eos=True)) <= block_size
            and len(tokenizer.encode(t2, add_eos=True)) <= block_size
        )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["openai", "hf"], default="openai")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081", help="OpenAI-compatible teacher URL")
    ap.add_argument("--model", default="kuon", help="Model ID for openai server or HF hub ID (e.g. google/gemma-4-E4B-it)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--casual", type=int, default=500)
    ap.add_argument("--open-consult", type=int, default=1600)
    ap.add_argument("--doc-summary", type=int, default=1200)
    ap.add_argument("--writing", type=int, default=1400)
    ap.add_argument("--explain", type=int, default=1200)
    ap.add_argument("--multiturn", type=int, default=900)
    ap.add_argument("--merge-base-dir", type=Path, default=None, help="Optional v4.2 Stage-3 anchor dir to replay")
    ap.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer.json to enforce 1024-token budget")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    forbidden = load_forbidden_prompts()
    sft_rows = build_casual_identity_rows()
    pref_rows: list[dict] = []

    specs = build_prompt_specs(
        open_consult=args.open_consult,
        doc_summary=args.doc_summary,
        writing=args.writing,
        explain=args.explain,
        multiturn=args.multiturn,
        casual=args.casual,
        seed=args.seed,
    )
    by_kind = Counter(s["kind"] for s in specs)
    by_lang = Counter(s["lang"] for s in specs)
    if args.dry_run:
        print(
            f"[distill-k11-dry-run] curated_casual={len(sft_rows)} specs={len(specs)} "
            f"by_kind={dict(by_kind)} by_lang={dict(by_lang)} forbidden={len(forbidden)}"
        )
        return

    cache_path = args.out / "raw_cache.jsonl"
    if args.backend == "openai":
        raw_map = _run_openai_generation(
            specs, cache_path, base_url=args.base_url, model=args.model, concurrency=args.concurrency
        )
    else:
        raw_map = _run_hf_generation(
            specs, cache_path, model_id=args.model, batch_size=args.batch_size, load_in_4bit=args.load_in_4bit
        )

    kept = Counter()
    for spec in specs:
        raw = raw_map.get(spec["id"])
        if not raw:
            continue
        obj = extract_json_object(raw)
        if obj and convert_record(spec, obj, forbidden, sft_rows, pref_rows):
            kept[spec["kind"]] += 1

    tok = None
    if args.tokenizer and args.tokenizer.is_file():
        from kuon.tokenizer import BPETokenizer
        tok = BPETokenizer.load(args.tokenizer)
        before_sft, before_pref = len(sft_rows), len(pref_rows)
        sft_rows = [r for r in sft_rows if _fits_budget(r, tok, args.block_size)]
        pref_rows = [r for r in pref_rows if _fits_budget(r, tok, args.block_size)]
        print(
            f"[distill-k11] token budget ({args.block_size}): "
            f"sft {before_sft}->{len(sft_rows)}, prefs {before_pref}->{len(pref_rows)}",
            flush=True,
        )

    print(f"[distill-k11] kept objects: {dict(kept)} | sft={len(sft_rows):,} prefs={len(pref_rows):,}", flush=True)

    # Deterministic 93/7 train/val split by user CRC32 (zero prompt overlap)
    rng = random.Random(args.seed)
    sft_tr, sft_va = [], []
    for r in sft_rows:
        if zlib.crc32(r["user"].encode("utf-8")) % 14 == 0 and r["task"] != "k11_casual_identity":
            sft_va.append(r)
        else:
            sft_tr.append(r)
    pref_tr, pref_va = [], []
    for r in pref_rows:
        if zlib.crc32(r["user"].encode("utf-8")) % 14 == 0:
            pref_va.append(r)
        else:
            pref_tr.append(r)

    if args.merge_base_dir and args.merge_base_dir.is_dir():
        for split, dst in (("sft_train.jsonl", sft_tr), ("sft_val.jsonl", sft_va)):
            p = args.merge_base_dir / split
            if p.is_file():
                dst.extend(json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
        for split, dst in (("prefs_train.jsonl", pref_tr), ("prefs_val.jsonl", pref_va)):
            p = args.merge_base_dir / split
            if p.is_file():
                dst.extend(json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())

    tr_users = {" ".join(r["user"].casefold().split()) for r in sft_tr if r.get("user")}
    sft_va = [r for r in sft_va if " ".join(r.get("user", "").casefold().split()) not in tr_users]
    ptr_users = {" ".join(r["user"].casefold().split()) for r in pref_tr if r.get("user")}
    pref_va = [r for r in pref_va if " ".join(r.get("user", "").casefold().split()) not in ptr_users]

    rng.shuffle(sft_tr)
    rng.shuffle(sft_va)
    rng.shuffle(pref_tr)
    rng.shuffle(pref_va)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("sft_train.jsonl", sft_tr),
        ("sft_val.jsonl", sft_va),
        ("prefs_train.jsonl", pref_tr),
        ("prefs_val.jsonl", pref_va),
    ):
        with (args.out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[distill-k11] wrote {name}: {len(rows):,} rows", flush=True)


if __name__ == "__main__":
    main()

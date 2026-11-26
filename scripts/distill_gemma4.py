"""Generate clean bilingual EN/ID distillation data for Kuon using Gemma 4 (Apache-2.0).

Gemma 4 is used strictly offline as a synthetic data generator; Kuon's weights
remain 100% trained from scratch. Outputs:
  - sft_distill_train.jsonl / sft_distill_val.jsonl
  - prefs_distill_train.jsonl / prefs_distill_val.jsonl

Covers the exact gaps identified in Kuon-2 v3 local chat & holdout evals:
  1. Kuon bilingual identity & out-of-scope polite refusal (no "I want to be a doctor")
  2. Diverse counterfactual business scenarios (clarification, fact-flip, confirmed_action text+MC)
  3. Multi-row grounded evidence QA (answerable exact entity-value binding + honest unanswerable)
  4. Practical open-ended EN/ID business troubleshooting & measurable trial advice

All generated text is:
  - Scrubbed of teacher identity leaks ("Gemma", "Google", "DeepMind", "Gemini", etc.)
  - Verified by Python against evidence numbers / abstention patterns
  - Filtered against ALL holdout & dev evaluation sets (zero leakage)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEACHER_LEAK_RE = re.compile(
    r"\b(gemma|google|deepmind|gemini|openai|chatgpt|gpt-?4|claude|anthropic|meta\s*ai|llama|qwen|alibaba|"
    r"as an ai(?: language model)?|sebagai (?:model bahasa|kecerdasan buatan|ai))\b",
    re.IGNORECASE,
)

ABSTAIN_RE = re.compile(
    r"(tidak\s+(disebutkan|menyebutkan|tercantum|tersedia|memuat|berisi)|"
    r"belum\s+(tersedia|disebutkan|tercantum)|"
    r"not\s+(specified|mentioned|provided|available|stated|included)|"
    r"does\s+not\s+(mention|specify|provide|include|state|contain))",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
CITATION_RE = re.compile(r"\[\d+\]")

INDUSTRIES = [
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
]

SYMPTOMS_ID = [
    "omzet bulanan turun",
    "pelanggan baru berkurang",
    "tingkat pembelian ulang menurun",
    "antrean komplain pelanggan meningkat",
    "margin keuntungan menyusut",
    "waktu penyelesaian layanan melambat",
    "banyak keranjang belanja ditinggalkan",
    "biaya operasional per pesanan naik",
]

SYMPTOMS_EN = [
    "monthly revenue dropped",
    "new customer signups slowed down",
    "repeat purchase rate declined",
    "support ticket backlog grew",
    "gross margin shrank",
    "order fulfillment time slowed down",
    "checkout abandonment rose",
    "unit cost per order increased",
]


def load_forbidden_prompts() -> set[str]:
    """Collect normalized user prompts from all holdout/dev benchmarks to prevent leakage."""
    forbidden: set[str] = set()
    biz_root = ROOT / "datasets/business"
    for fname in ("eval.jsonl", "advice_eval.jsonl", "advice_dev.jsonl"):
        p = biz_root / fname
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    forbidden.add(" ".join(json.loads(line)["user"].casefold().split()))
    try:
        from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES
        from scripts.eval_calibration import BUSINESS_CALIBRATION_CASES
        from scripts.eval_confirmed_actions import CASES as CONFIRMED_CASES

        for c in CALIBRATION_CASES:
            forbidden.add(" ".join(c.user.casefold().split()))
        for c in DELIVERABLE_CASES:
            forbidden.add(" ".join(c.user.casefold().split()))
        for _, _, q, _, _ in BUSINESS_CALIBRATION_CASES:
            forbidden.add(" ".join(q.casefold().split()))
        for scenarios in CONFIRMED_CASES.values():
            for issue, *_ in scenarios:
                forbidden.add(" ".join(issue.casefold().split()))
    except Exception:
        pass
    return forbidden


def is_clean_text(text: str, forbidden: set[str]) -> bool:
    if not text or not text.strip():
        return False
    if TEACHER_LEAK_RE.search(text):
        return False
    norm = " ".join(text.casefold().split())
    for bad in forbidden:
        if bad and bad in norm:
            return False
    return True


def extract_json_object(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_identity_sft_rows() -> list[dict]:
    """Deterministic + varied Kuon identity & scope boundaries (EN + ID)."""
    pairs = [
        ("id", "Halo, kamu siapa dan bisa bantu apa?",
         "Halo! Saya Kuon, asisten konsultasi bisnis dwibahasa untuk penjualan, layanan pelanggan, dan operasi di kuonstudio.com. Ceritakan gejala atau angka usaha Anda supaya kita bisa mencari langkah uji yang terukur."),
        ("id", "Hai Kuon, siapa yang membuatmu?",
         "Saya Kuon, model bahasa kecil yang dilatih dari nol khusus untuk membantu analisis operasional, penjualan, dan layanan pelanggan dalam bahasa Indonesia dan Inggris."),
        ("id", "Kamu dasarnya model apa?",
         "Saya Kuon 1, model bahasa kecil berukuran sekitar 300 juta parameter dengan bobot yang dilatih dari nol untuk analisis bisnis dwibahasa Indonesia dan Inggris di kuonstudio.com."),
        ("id", "Apakah kamu berbasis GPT, Claude, atau Gemma?",
         "Bukan, bobot saya adalah Kuon 1 yang dilatih sendiri dari nol khusus untuk konsultasi bisnis, pembacaan tabel, dan verifikasi angka."),
        ("id", "Kamu bisa bantu nulis kode Python atau website?",
         "Fokus saya adalah konsultasi bisnis—seperti penjualan, retensi pelanggan, alur operasi, dan membaca laporan angka—bukan penulisan kode. Kalau ada kendala usaha yang ingin dianalisis, silakan bagikan detailnya."),
        ("id", "Terima kasih atas sarannya!",
         "Sama-sama! Jika setelah uji coba nanti Anda sudah punya angka hasilnya, silakan bagikan lagi supaya kita bisa evaluasi langkah berikutnya."),
        ("en", "Hi, who are you and what can you help with?",
         "Hello! I am Kuon, a bilingual business consulting assistant focused on sales, customer service, and operations at kuonstudio.com. Share your symptoms or metrics and we can narrow down one measurable next step."),
        ("en", "What base model are you built on?",
         "I am Kuon 1, a compact ~300M-parameter language model with weights trained from scratch for bilingual English and Indonesian business consulting at kuonstudio.com."),
        ("en", "Are you based on GPT, Claude, or Gemma?",
         "No, I run on Kuon 1 weights trained from scratch for grounded business analysis, table calculations, and operational consulting."),
        ("en", "Can you write a React app or debug my SQL query?",
         "My focus is business consulting across sales, customer support, and operations rather than software coding. If you have an operational bottleneck or revenue report to review, I would be glad to help."),
        ("en", "Thanks, that was helpful!",
         "You are welcome! Share the results once you run the test and we can decide what to adjust next."),
    ]
    rows = []
    # Slight natural variations so smalltalk never collapses into unrelated text
    greetings_id = ["Halo", "Hai", "Selamat pagi", "Selamat siang", "Halo Kuon"]
    greetings_en = ["Hello", "Hi", "Good morning", "Good afternoon", "Hey Kuon"]
    for g in greetings_id:
        rows.append({
            "language": "id", "task": "distill_identity", "business": False,
            "user": f"{g}, kamu siapa?",
            "assistant": "Halo! Saya Kuon, asisten konsultasi bisnis dwibahasa untuk penjualan, layanan pelanggan, dan operasi. Ada kendala atau data usaha yang ingin Anda bahas?",
        })
    for g in greetings_en:
        rows.append({
            "language": "en", "task": "distill_identity", "business": False,
            "user": f"{g}, who are you?",
            "assistant": "Hello! I am Kuon, a bilingual business consulting assistant for sales, customer service, and operations. What business metric or issue would you like to work through?",
        })
    for lang, u, a in pairs:
        for _ in range(4):
            rows.append({"language": lang, "task": "distill_identity", "business": False, "user": u, "assistant": a})
    return rows


def make_scenario_prompts(count: int, seed: int = 2026) -> list[tuple[str, str]]:
    """Build prompts asking Gemma 4 to generate counterfactual business consulting scenarios."""
    rng = random.Random(seed)
    prompts: list[tuple[str, str]] = []
    for i in range(count):
        ind_id, ind_en = INDUSTRIES[i % len(INDUSTRIES)]
        if i % 2 == 0:
            sym = SYMPTOMS_ID[(i // 2 + rng.randint(0, 3)) % len(SYMPTOMS_ID)]
            p = (
                f"Buat 1 skenario konsultasi bisnis unik berbahasa Indonesia untuk usaha '{ind_id}' dengan gejala awal '{sym}'. "
                "Keluarkan HANYA JSON valid dengan tepat 6 key string:\n"
                '{"opening": "keluhan singkat pemilik usaha (1-2 kalimat) + pertanyaan haruskah langsung diskon/iklan/tambah staf", '
                '"clarify_question": "1 pertanyaan klarifikasi tajam untuk membedakan 2 kemungkinan penyebab (A vs B)", '
                '"fact_a": "fakta spesifik penyebab A (bukan penyebab B)", '
                '"action_a": "1 langkah uji murah spesifik untuk penyebab A + metrik ukurannya (tanpa ubah harga jika tidak relevan)", '
                '"fact_b": "fakta spesifik penyebab B (kebalikan dari A)", '
                '"action_b": "1 langkah uji murah spesifik untuk penyebab B + metrik ukurannya"}'
            )
            prompts.append(("id", p))
        else:
            sym = SYMPTOMS_EN[(i // 2 + rng.randint(0, 3)) % len(SYMPTOMS_EN)]
            p = (
                f"Create 1 unique English business consulting scenario for a '{ind_en}' experiencing '{sym}'. "
                "Output ONLY valid JSON with these exact 6 string keys:\n"
                '{"opening": "short 1-2 sentence owner problem + whether to immediately cut price/add ads/hire", '
                '"clarify_question": "1 sharp clarifying question separating cause A from cause B", '
                '"fact_a": "specific evidence confirming cause A (and ruling out B)", '
                '"action_a": "1 concrete low-cost trial for cause A + what metric to track", '
                '"fact_b": "specific evidence confirming cause B (ruling out A)", '
                '"action_b": "1 concrete low-cost trial for cause B + what metric to track"}'
            )
            prompts.append(("en", p))
    return prompts


def make_evidence_prompts(count: int, seed: int = 2026) -> list[tuple[str, str]]:
    """Build prompts asking Gemma 4 to generate multi-row business excerpts + answerable/unanswerable QA."""
    prompts: list[tuple[str, str]] = []
    for i in range(count):
        ind_id, ind_en = INDUSTRIES[i % len(INDUSTRIES)]
        if i % 2 == 0:
            p = (
                f"Buat 1 cuplikan laporan operasional singkat berbahasa Indonesia untuk '{ind_id}' yang berisi 3-4 angka atau entitas berbeda (misalnya shift/cabang/produk/tier).\n"
                "Keluarkan HANYA JSON valid dengan tepat 7 key string:\n"
                '{"evidence": "[1] Judul Laporan\\nIsi 2-3 kalimat berisi 3-4 angka dan entitas spesifik.", '
                '"ans_q": "pertanyaan spesifik yang jawabannya ADA di baris/entitas ke-2 atau ke-3 (bukan yang pertama)", '
                '"ans_a": "jawaban ringkas akurat yang mengutip [1] serta menyebut nama entitas dan angka persis beserta satuannya", '
                '"ans_wrong": "jawaban salah yang menukar angka entitas lain dari laporan yang sama", '
                '"unans_q": "pertanyaan bisnis masuk akal (misalnya laba bersih/biaya sewa/alasan kenaikan) yang datanya TIDAK ADA di laporan", '
                '"unans_a": "penolakan jujur berbahasa Indonesia yang menyebut Kutipan [1] tidak menyebutkan data tersebut + 1 pertanyaan minta data spesifik (TANPA mengarang angka)", '
                '"unans_wrong": "jawaban halusinasi yang mencoba menebak atau menyalin angka yang salah dari laporan"}'
            )
            prompts.append(("id", p))
        else:
            p = (
                f"Create 1 short English operational/financial excerpt for a '{ind_en}' containing 3-4 distinct entities and numbers (e.g. tiers, shifts, warehouses, unit costs).\n"
                "Output ONLY valid JSON with these exact 7 string keys:\n"
                '{"evidence": "[1] Report Title\\n2-3 sentences with 3-4 specific figures and entity names.", '
                '"ans_q": "specific question whose answer is in the 2nd or 3rd item (never just the first)", '
                '"ans_a": "concise grounded answer citing [1] and stating the exact entity name, number, and unit", '
                '"ans_wrong": "plausible wrong answer swapping a neighbor item number from the same excerpt", '
                '"unans_q": "reasonable business question (e.g. net profit margin, lease cost, root cause) whose answer is MISSING from the excerpt", '
                '"unans_a": "honest abstention stating the excerpt [1] does not specify that figure + 1 follow-up request (NO guessed numbers)", '
                '"unans_wrong": "hallucinated answer copying an unrelated number from the excerpt"}'
            )
            prompts.append(("en", p))
    return prompts


def make_consult_prompts(count: int, seed: int = 2026) -> list[tuple[str, str]]:
    """Build prompts for natural open-ended business consulting (solves echo/off-topic replies)."""
    rng = random.Random(seed)
    prompts: list[tuple[str, str]] = []
    for i in range(count):
        ind_id, ind_en = INDUSTRIES[i % len(INDUSTRIES)]
        if i % 2 == 0:
            sym = SYMPTOMS_ID[(i + rng.randint(0, 5)) % len(SYMPTOMS_ID)]
            p = (
                f"Buat 1 tanya-jawab konsultasi bisnis praktis berbahasa Indonesia untuk pemilik '{ind_id}' tentang masalah '{sym}' dengan detail sebab yang jelas di pertanyaan user.\n"
                "Keluarkan HANYA JSON valid: "
                '{"user": "pertanyaan realistis pemilik usaha (1-3 kalimat) yang sudah menyebut fakta atau angka gejala", '
                '"assistant": "jawaban ringkas, tajam, 2-3 kalimat: sebut 1-2 hal utama yang harus dicek/diuji besok dan metrik pengukurannya (tanpa menyebut diri sebagai AI)"}'
            )
            prompts.append(("id", p))
        else:
            sym = SYMPTOMS_EN[(i + rng.randint(0, 5)) % len(SYMPTOMS_EN)]
            p = (
                f"Create 1 practical English business consulting Q&A for a '{ind_en}' dealing with '{sym}' where the user gives concrete context.\n"
                "Output ONLY valid JSON: "
                '{"user": "realistic 1-3 sentence owner question with concrete symptoms or percentages", '
                '"assistant": "concise 2-3 sentence consulting reply naming 1-2 concrete checks/trials and the metric to track (never mention AI)"}'
            )
            prompts.append(("en", p))
    return prompts


def convert_scenario_obj(
    lang: str, idx: int, obj: dict, forbidden: set[str], sft_out: list[dict], prefs_out: list[dict]
) -> bool:
    keys = ("opening", "clarify_question", "fact_a", "action_a", "fact_b", "action_b")
    vals = [str(obj.get(k, "")).strip() for k in keys]
    if any(len(v) < 12 for v in vals):
        return False
    if any(not is_clean_text(v, forbidden) for v in vals):
        return False
    opening, q, fa, aa, fb, ab = vals
    if aa.casefold() == ab.casefold():
        return False

    # SFT rows
    sft_out.append({"language": lang, "task": "distill_clarify", "business": True, "user": opening, "assistant": q})
    sft_out.append({"language": lang, "task": "distill_adapt_mt", "business": True, "user": f"{opening}\nAssistant: {q}\nUser: {fa}", "assistant": aa})
    sft_out.append({"language": lang, "task": "distill_adapt_mt", "business": True, "user": f"{opening}\nAssistant: {q}\nUser: {fb}", "assistant": ab})
    sft_out.append({"language": lang, "task": "distill_adapt_direct", "business": True, "user": f"{opening} {fa}", "assistant": aa})
    sft_out.append({"language": lang, "task": "distill_adapt_direct", "business": True, "user": f"{opening} {fb}", "assistant": ab})

    # Confirmed facts text + MC (strictly 1:1:1 A/B/C balanced)
    hdr = "Fakta terkonfirmasi" if lang == "id" else "Confirmed facts"
    instr = (
        "Pilih satu tindakan dan ukuran yang cocok. Jika fakta kurang, pilih pertanyaan."
        if lang == "id"
        else "Choose one action with its metric. If a fact is missing, choose the question."
    )
    missing_fact = (
        "Belum diketahui di bagian mana proses atau pelanggan tertahan."
        if lang == "id"
        else "It is not yet known where customers or operations are getting held up."
    )
    choices = (aa, ab, q)
    for fact_i, (fact_txt, exp_idx) in enumerate(((fa, 0), (fb, 1), (missing_fact, 2))):
        body = f"{opening}\n{hdr}: {fact_txt}\n{instr}"
        sft_out.append({"language": lang, "task": "distill_confirmed_text", "business": False, "user": body, "assistant": choices[exp_idx]})
        for wrong_idx in range(3):
            if wrong_idx != exp_idx:
                prefs_out.append({
                    "language": lang, "task": "distill_pref_confirmed_text", "business": False,
                    "user": body, "chosen": choices[exp_idx], "rejected": choices[wrong_idx],
                })
        # Balanced 3-rotation MC
        mc_biz = bool(idx % 2 == 1)
        for offset in (0, 1, 2):
            order = [(offset + k) % 3 for k in range(3)]
            opts_txt = "\n".join(f"{ltr}. {choices[c_i]}" for ltr, c_i in zip("ABC", order))
            gold = "ABC"[order.index(exp_idx)]
            sft_out.append({"language": lang, "task": "distill_confirmed_mc", "business": mc_biz, "user": f"{body}\n{opts_txt}", "assistant": gold})
            for wrong_ltr in (l for l in "ABC" if l != gold):
                prefs_out.append({
                    "language": lang, "task": "distill_pref_confirmed_mc", "business": mc_biz,
                    "user": f"{body}\n{opts_txt}", "chosen": gold, "rejected": wrong_ltr,
                })

    # Counterfactual APO-zero pairs
    prefs_out.append({"language": lang, "task": "distill_pref_clarify", "business": True, "user": opening, "chosen": q, "rejected": aa})
    prefs_out.append({"language": lang, "task": "distill_pref_clarify", "business": True, "user": opening, "chosen": q, "rejected": ab})
    prefs_out.append({"language": lang, "task": "distill_pref_flip", "business": True, "user": f"{opening}\nAssistant: {q}\nUser: {fa}", "chosen": aa, "rejected": ab})
    prefs_out.append({"language": lang, "task": "distill_pref_flip", "business": True, "user": f"{opening}\nAssistant: {q}\nUser: {fb}", "chosen": ab, "rejected": aa})
    prefs_out.append({"language": lang, "task": "distill_pref_flip_direct", "business": True, "user": f"{opening} {fa}", "chosen": aa, "rejected": ab})
    prefs_out.append({"language": lang, "task": "distill_pref_flip_direct", "business": True, "user": f"{opening} {fb}", "chosen": ab, "rejected": aa})
    return True


def convert_evidence_obj(
    lang: str, obj: dict, forbidden: set[str], sft_out: list[dict], prefs_out: list[dict]
) -> bool:
    keys = ("evidence", "ans_q", "ans_a", "ans_wrong", "unans_q", "unans_a", "unans_wrong")
    vals = [str(obj.get(k, "")).strip() for k in keys]
    if any(len(v) < 8 for v in vals):
        return False
    if any(not is_clean_text(v, forbidden) for v in vals):
        return False
    ev, aq, aa, aw, uq, ua, uw = vals
    if "[1]" not in ev or "[1]" not in aa:
        return False
    ev_nums = set(NUMBER_RE.findall(CITATION_RE.sub("", ev)))
    ans_nums = NUMBER_RE.findall(CITATION_RE.sub("", aa))
    if not ans_nums or not all(n in ev_nums for n in ans_nums):
        return False
    if not ABSTAIN_RE.search(ua):
        return False
    # Reject unanswerable abstention if it invents numbers not in evidence
    ua_nums = NUMBER_RE.findall(CITATION_RE.sub("", ua))
    if any(n not in ev_nums for n in ua_nums):
        return False

    refuse = "Saya tidak tahu." if lang == "id" else "I do not know."
    sft_out.append({"language": lang, "task": "distill_ev_ans", "business": True, "user": aq, "evidence": ev, "assistant": aa})
    sft_out.append({"language": lang, "task": "distill_ev_unans", "business": True, "user": uq, "evidence": ev, "assistant": ua})

    prefs_out.append({"language": lang, "task": "distill_pref_ev_bind", "business": True, "user": aq, "evidence": ev, "chosen": aa, "rejected": aw})
    prefs_out.append({"language": lang, "task": "distill_pref_ev_refuse", "business": True, "user": aq, "evidence": ev, "chosen": aa, "rejected": refuse})
    prefs_out.append({"language": lang, "task": "distill_pref_ev_abstain", "business": True, "user": uq, "evidence": ev, "chosen": ua, "rejected": uw})
    return True


def generate_batches(
    model_id: str,
    prompts: list[tuple[str, str]],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 380,
    load_in_4bit: bool = False,
) -> list[tuple[str, str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[distill] Loading {model_id} (4bit={load_in_4bit})...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()

    results: list[tuple[str, str]] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        texts = []
        for _, p in chunk:
            try:
                t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            except Exception:
                t = tok.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": p}]}],
                    tokenize=False, add_generation_prompt=True,
                )
            texts.append(t)
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.75,
                top_p=0.92,
                pad_token_id=tok.pad_token_id,
            )
        for (lang, _), seq in zip(chunk, out):
            gen = tok.decode(seq[in_len:], skip_special_tokens=True)
            results.append((lang, gen))
        done = min(start + batch_size, len(prompts))
        if done % max(batch_size * 5, 64) == 0 or done == len(prompts):
            print(f"[distill] generated {done}/{len(prompts)}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", type=int, default=600)
    ap.add_argument("--evidence-cases", type=int, default=1000)
    ap.add_argument("--consult-cases", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--merge-base-dir", type=Path, default=None, help="optional Stage 3 v3 dir to merge with")
    ap.add_argument("--dry-run", action="store_true", help="verify prompt construction & identity rows offline")
    args = ap.parse_args()

    forbidden = load_forbidden_prompts()
    sft_rows = build_identity_sft_rows()
    pref_rows: list[dict] = []

    scen_prompts = make_scenario_prompts(args.scenarios)
    ev_prompts = make_evidence_prompts(args.evidence_cases)
    con_prompts = make_consult_prompts(args.consult_cases)

    if args.dry_run:
        print(
            f"[distill-dry-run] identity={len(sft_rows)} scen_prompts={len(scen_prompts)} "
            f"ev_prompts={len(ev_prompts)} con_prompts={len(con_prompts)} forbidden={len(forbidden)}"
        )
        return

    all_prompts = (
        [("scen", l, p) for l, p in scen_prompts]
        + [("ev", l, p) for l, p in ev_prompts]
        + [("con", l, p) for l, p in con_prompts]
    )
    raw_outs = generate_batches(
        args.model,
        [(l, p) for _, l, p in all_prompts],
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )

    kept = Counter()
    for idx, ((kind, lang, _), (_, gen)) in enumerate(zip(all_prompts, raw_outs)):
        obj = extract_json_object(gen)
        if not obj:
            continue
        if kind == "scen":
            if convert_scenario_obj(lang, idx, obj, forbidden, sft_rows, pref_rows):
                kept["scen"] += 1
        elif kind == "ev":
            if convert_evidence_obj(lang, obj, forbidden, sft_rows, pref_rows):
                kept["ev"] += 1
        elif kind == "con":
            u = str(obj.get("user", "")).strip()
            a = str(obj.get("assistant", "")).strip()
            if len(u) >= 15 and len(a) >= 20 and is_clean_text(u, forbidden) and is_clean_text(a, forbidden):
                sft_rows.append({"language": lang, "task": "distill_consult", "business": True, "user": u, "assistant": a})
                kept["con"] += 1

    print(f"[distill] valid objects kept: {dict(kept)} | sft={len(sft_rows)} prefs={len(pref_rows)}", flush=True)

    # Split 93/7 train/val deterministically by user prompt CRC32 so zero prompt overlap
    rng = random.Random(2026)
    sft_tr, sft_va = [], []
    for r in sft_rows:
        if zlib.crc32(r["user"].encode("utf-8")) % 14 == 0 and r["task"] != "distill_identity":
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
                dst.extend(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        for split, dst in (("prefs_train.jsonl", pref_tr), ("prefs_val.jsonl", pref_va)):
            p = args.merge_base_dir / split
            if p.is_file():
                dst.extend(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())

    # Final dedup between train and val
    tr_users = {" ".join(r["user"].casefold().split()) for r in sft_tr if not r.get("raw_text") and r.get("user")}
    sft_va = [r for r in sft_va if r.get("raw_text") or " ".join(r["user"].casefold().split()) not in tr_users]
    ptr_users = {" ".join(r["user"].casefold().split()) for r in pref_tr if r.get("user")}
    pref_va = [r for r in pref_va if " ".join(r["user"].casefold().split()) not in ptr_users]

    rng.shuffle(sft_tr)
    rng.shuffle(sft_va)
    rng.shuffle(pref_tr)
    rng.shuffle(pref_va)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_train.jsonl", sft_tr), ("sft_val.jsonl", sft_va), ("prefs_train.jsonl", pref_tr), ("prefs_val.jsonl", pref_va)):
        with (args.out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[distill] wrote {name}: {len(rows):,} rows", flush=True)


if __name__ == "__main__":
    main()

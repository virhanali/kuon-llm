"""Build Kuon 0.1 Stage 3A Clean Diverse Bilingual Business SFT + Experience Rehearsal corpus.

Designed directly from the failure analysis of Stage 3 v1 (train 0.42 vs val 1.40 overfitting
and 272/300 over-abstention collapse) and 2024-2026 sub-billion SLM research (MobileLLM-R1,
SmolLM2/3, Abstain-R1, Context-DPO):
  - ZERO extreme duplication of narrow templates (advice_pairs is 1x, NOT 33x).
  - Diverse, license-verified EN + ID datasets so a ~120M model learns general instruction
    following, reading comprehension, clarification, and grounded business reasoning rather
    than memorizing 24 fixed sentences.
  - Abstention ratio capped (~12-15% overall) and paired with post-refusal clarification
    questions (Abstain-R1) instead of a 720-row bare "I do not know." shortcut.
  - ~15% Bilingual Pretraining Experience Rehearsal (`raw_text: True`) from FineWeb2-HQ
    Indonesian + OpenStax Business/Finance textbooks to prevent catastrophic forgetting.

All sources are commercially open-licensed (Apache-2.0, MIT, CC-BY-4.0, CC-BY-SA, ODC-By,
CDLA-Permissive/Sharing) and 100% disjoint from all holdout evaluation files.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from kuon.chat import build_example
from kuon.tokenizer import BPETokenizer
from scripts.prepare_balanced_business import SCENARIOS, examples as scenario_examples
from scripts.prepare_kuon1_corpus import (
    OPENSTAX_BUSINESS_FILES,
    OPENSTAX_REV,
    HttpRangeFile,
    USER_AGENT,
    clean_text,
    hf_shard_url,
)

ROOT = Path(__file__).resolve().parents[1]

IRRELEVANT_NOTES = {
    "id": (
        "Catatan ini ditulis saat rapat pagi.",
        "Berkas laporan disimpan di folder bersama.",
        "Warna sampul buku catatan toko adalah biru.",
    ),
    "en": (
        "These notes were typed after the morning meeting.",
        "The report file is stored in the shared folder.",
        "The store notebook has a blue cover.",
    ),
}

BITEXT_PLACEHOLDERS = {
    "{{Order Number}}": "#ORD-4821",
    "{{Invoice Number}}": "#INV-2094",
    "{{Account Type}}": "Business Pro",
    "{{Account Category}}": "Business",
    "{{Delivery City}}": "Bandung",
    "{{Delivery Country}}": "Indonesia",
    "{{Currency Symbol}}": "$",
    "{{Refund Amount}}": "45.00",
    "{{Person Name}}": "Customer Support",
    "{{Customer Support Phone Number}}": "1-800-555-0199",
    "{{Customer Support Hours}}": "Monday to Friday, 09:00-18:00",
    "{{Website URL}}": "our help portal",
    "{{Online Company Portal Info}}": "your account dashboard",
    "{{Online Order Interaction}}": "Order History",
    "{{Client First Name}}": "Alex",
    "{{Client Last Name}}": "Tan",
    "{{Company Name}}": "our store",
    "{{Date Range}}": "3-5 business days",
}


def _resolve_url(url: str) -> str:
    req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as r:
        return r.geturl()


def _fill_placeholders(text: str) -> str:
    out = text
    for k, v in BITEXT_PLACEHOLDERS.items():
        out = out.replace(k, v)
    return out


def build_counterfactual_advice_rows() -> tuple[list[dict], list[dict]]:
    """1x clean counterfactual business advice rows (NO 3x upsampling!).

    Preserves the exact RNG seed (1337) and 24-train / 6-dev split of
    `prepare_advice_pairs.py` so `advice_dev` is strictly validation-only.
    """
    biz_root = ROOT / "datasets/business"
    groups: dict[str, list[tuple[str, str, str, str, str, str, str]]] = defaultdict(list)
    for line in (biz_root / "advice_pairs.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        lang, name, opening, question, fact_a, answer_a, fact_b, answer_b = line.split("|")
        groups[lang].append((name, opening, question, fact_a, answer_a, fact_b, answer_b))

    rng = random.Random(1337)
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for lang in ("id", "en"):
        cases = list(groups[lang])
        assert len(cases) == 30
        rng.shuffle(cases)
        for is_val, subset in ((False, cases[:24]), (True, cases[24:])):
            target = val_rows if is_val else train_rows
            for idx, (name, opening, question, fact_a, answer_a, fact_b, answer_b) in enumerate(subset):
                meta = {"language": lang, "scenario": f"{lang}:{name}", "business": True}
                target.append({**meta, "task": "clarification", "user": opening, "assistant": question})
                notes = IRRELEVANT_NOTES[lang]
                for f_idx, (fact, answer) in enumerate(((fact_a, answer_a), (fact_b, answer_b))):
                    target.append({
                        **meta, "task": "adaptation_multiturn",
                        "user": f"{opening}\nAssistant: {question}\nUser: {fact}",
                        "assistant": answer,
                    })
                    if not is_val:
                        note = notes[(idx + f_idx) % len(notes)]
                        target.append({
                            **meta, "task": "adaptation_direct",
                            "user": f"{opening} {fact}", "assistant": answer,
                        })
                        target.append({
                            **meta, "task": "adaptation_invariance",
                            "user": f"{opening} {fact} {note}", "assistant": answer,
                        })
                        target.append({
                            **meta, "task": "confirmed_action_evidence",
                            "user": opening,
                            "evidence": f"[1] {fact}\n[2] {note}",
                            "assistant": answer,
                        })
                # Multiple-choice & confirmed-fact action selection (permutation-invariant A/B/C + irrelevant date)
                # Built strictly from the 24+24 train split (dev split goes only to val_rows).
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
                date_note = (
                    ("Catatan dicatat hari Selasa." if idx % 2 == 0 else "Catatan dicatat hari Rabu.")
                    if lang == "id"
                    else ("The note was logged on Tuesday." if idx % 2 == 0 else "The note was logged on Wednesday.")
                )
                choices = (answer_a, answer_b, question)
                mc_items = ((fact_a, 0), (fact_b, 1), (missing_fact, 2))
                for fact_i, (confirmed, exp_idx) in enumerate(mc_items):
                    body = f"{opening}\n{hdr}: {confirmed} {date_note}\n{instr}"
                    # Plain text selection with Confirmed facts header (balanced subset, ~72 train rows)
                    if not is_val and idx % 2 == 0:
                        target.append({
                            **meta, "task": "confirmed_action_text",
                            "business": False, "user": body, "assistant": choices[exp_idx],
                        })
                    # Balanced cyclic offset: within EVERY scenario the 3 facts map to distinct {A, B, C}
                    offset = (idx + 2 * fact_i) % 3
                    order = [(offset + k) % 3 for k in range(3)]
                    opts_txt = "\n".join(f"{ltr}. {choices[c_i]}" for ltr, c_i in zip("ABC", order))
                    gold_letter = "ABC"[order.index(exp_idx)]
                    target.append({
                        **meta, "task": "confirmed_action_mc",
                        "business": False,
                        "user": f"{body}\n{opts_txt}",
                        "assistant": gold_letter,
                    })

    # Concise direct-fix operational consulting rows (train-only, prevents echoing user questions)
    direct_fixes = [
        ("id", "Dua kiriman terakhir salah alamat karena label dicetak manual tanpa cek kode pos. Apa perbaikan cepat besok?",
         "Wajibkan pemeriksaan kode pos sebelum label dicetak dan catat jumlah kiriman salah alamat setiap hari."),
        ("id", "Jadwal stok opname gudang sering molor karena tidak ada petugas yang memegang daftar akhir. Apa langkah murah yang bisa dicoba?",
         "Tunjuk satu penanggung jawab daftar akhir dengan batas jam tetap dan ukur persentase opname yang selesai tepat waktu."),
        ("id", "Banyak pelanggan salah ambil minuman di meja ambil karena gelas tidak diberi nomor struk. Apa yang harus diuji besok?",
         "Tulis nomor struk di setiap gelas dan cek nomor struk saat serah terima, lalu catat jumlah pesanan tertukar harian."),
        ("en", "Two recent shipments went to the wrong branch because labels were typed without checking branch codes. What is a low-cost fix?",
         "Require a branch-code check before printing each label and track misrouted shipments daily."),
        ("en", "Our daily handoff checklist is often skipped because nobody owns the closing sign-off. What should we try tomorrow?",
         "Assign one owner and a fixed sign-off time each shift, then track the percentage of shifts completed on time."),
        ("en", "Customers keep grabbing the wrong pickup bags because staff only call first names. What is a simple fix to test?",
         "Verify the order number on the receipt before handing over each bag and log daily mix-up incidents."),
    ]
    for lang, u, a in direct_fixes:
        train_rows.append({"language": lang, "task": "operational_direct_fix", "business": True, "user": u, "assistant": a})
    return train_rows, val_rows


def build_grounded_business_rows() -> tuple[list[dict], list[dict]]:
    """1x grounded evidence QA, abstention, rewrite, and summary rows from scenarios.tsv."""
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    split = train_rows
    index = 0
    for line in SCENARIOS.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Development"):
            split = val_rows
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        for language, section in (("en", parts[:4]), ("id", parts[4:])):
            for row in scenario_examples(section, index, language):
                split.append({**row, "language": language, "business": True})
        index += 1

    # Compact, balanced numeric business calibration drills (60 train + 12 val per lang, NOT 400!)
    cities_id = ["Bandung", "Medan", "Semarang", "Makassar", "Yogyakarta", "Denpasar", "Malang", "Bogor"]
    cities_en = ["Austin", "Seattle", "Denver", "Boston", "Chicago", "Portland", "Atlanta", "Miami"]
    items_id = ["Kopi Susu", "Roti Bakar", "Kemeja Linen", "Sepatu Kulit", "Tas Ransel", "Sabun Herbal"]
    items_en = ["Espresso Maker", "Desk Lamp", "Ceramic Mug", "Travel Backpack", "Cotton Hoodie", "Water Filter"]

    for i in range(36):
        target = val_rows if i >= 30 else train_rows
        c_id = cities_id[i % len(cities_id)]
        it_id = items_id[i % len(items_id)]
        pct = 12 + (i * 7) % 65
        rev = 45 + (i * 11) % 150
        orders = 300 + (i * 29) % 900
        target.append({
            "language": "id", "task": "calibration_answerable", "business": True,
            "user": f"Berapa persen pesanan di cabang {c_id} yang berasal dari produk {it_id}?",
            "evidence": f"[1] Rekap Cabang {c_id}\nTotal transaksi: {orders} pesanan. Produk {it_id} menyumbang {pct}% dari seluruh pesanan minggu ini.",
            "assistant": f"Berdasarkan rekap cabang {c_id} [1], produk {it_id} menyumbang {pct}% dari total {orders} pesanan.",
        })
        target.append({
            "language": "id", "task": "calibration_unanswerable", "business": True,
            "user": f"Berapa laba bersih setelah pajak di cabang {c_id} minggu ini?",
            "evidence": f"[1] Ringkasan Omzet {c_id}\nCabang {c_id} mencatat omzet kotor Rp{rev} juta dari {orders} transaksi minggu ini.",
            "assistant": f"Laba bersih setelah pajak cabang {c_id} tidak tercantum di ringkasan [1] (hanya tercatat omzet kotor Rp{rev} juta dari {orders} transaksi). Berapa total biaya operasional dan beban pajak cabang tersebut?",
        })
        c_en = cities_en[i % len(cities_en)]
        it_en = items_en[i % len(items_en)]
        target.append({
            "language": "en", "task": "calibration_answerable", "business": True,
            "user": f"What percentage of total orders at the {c_en} store came from {it_en}?",
            "evidence": f"[1] {c_en} Store Weekly Report\nTotal volume: {orders} orders. {it_en} accounted for {pct}% of all orders this week.",
            "assistant": f"According to the {c_en} Store Weekly Report [1], {it_en} accounted for {pct}% of the {orders} total orders.",
        })
        target.append({
            "language": "en", "task": "calibration_unanswerable", "business": True,
            "user": f"What was the net operating profit margin of the {c_en} store this week?",
            "evidence": f"[1] {c_en} Store Revenue Log\nThe {c_en} store recorded ${rev},000 in gross revenue across {orders} orders this week.",
            "assistant": f"The net operating profit margin is not specified in the report [1] (it only lists ${rev},000 in gross revenue across {orders} orders). Could you share the operating expenses and cost of goods sold?",
        })

    # Multi-entity selective extraction, min/max comparison, distractor-number restraint,
    # and numeric-context decision clarification (100% disjoint entities/domains from eval sets).
    triples_id = [
        ("Depo Bekasi", "Depo Cikarang", "Depo Karawang"),
        ("Regu A", "Regu B", "Regu C"),
        ("Vendor Orion", "Vendor Vega", "Vendor Nova"),
        ("Layanan Reguler", "Layanan Kilat", "Layanan Kargo"),
        ("Kategori Elektronik", "Kategori Aksesoris", "Kategori Peralatan"),
        ("Shift Pagi", "Shift Siang", "Shift Sore"),
    ]
    triples_en = [
        ("Central Depot", "Harbor Depot", "Valley Depot"),
        ("Team Alpha", "Team Beta", "Team Gamma"),
        ("Plan Basic", "Plan Growth", "Plan Scale"),
        ("Channel Email", "Channel Social", "Channel Partner"),
        ("Line A", "Line B", "Line C"),
        ("Tier Bronze", "Tier Silver", "Tier Gold"),
    ]
    for i in range(36):
        target = val_rows if i >= 30 else train_rows
        a_id, b_id, c_id = triples_id[i % len(triples_id)]
        v1 = 4 + (i * 3) % 15
        v2 = 11 + (i * 5) % 27
        v3 = 7 + (i * 7) % 19
        if v2 == v1:
            v2 += 3
        if v3 in (v1, v2):
            v3 += 5
        vol = 420 + (i * 37) % 800
        # 1. ID multi-entity selective lookup (ask 2nd or 3rd entity, never blindly 1st)
        ask_b = (i % 2 == 0)
        chosen_ent, chosen_val = (b_id, v2) if ask_b else (c_id, v3)
        distractor_val = v1
        target.append({
            "language": "id", "task": "grounded_multi_entity", "business": True,
            "user": f"Berapa hari waktu proses rata-rata untuk {chosen_ent} berdasarkan laporan operasional?",
            "evidence": (
                f"[1] Ringkasan Operasional Kuartal {(i % 4) + 1}\n"
                f"Total kiriman: {vol} unit. {a_id}: {v1} hari. {b_id}: {v2} hari. {c_id}: {v3} hari."
            ),
            "assistant": f"Berdasarkan Ringkasan Operasional [1], waktu proses rata-rata untuk {chosen_ent} adalah {chosen_val} hari.",
            "distractor": f"Berdasarkan Ringkasan Operasional [1], jawabannya adalah: {distractor_val} hari.",
        })
        # 2. ID argmax/argmin comparison across 3 items (diverse product/menu/branch phrasing)
        cmp_sets_id = [
            ("Produk", "Es Cokelat", "Croissant Mentega", "Jus Mangga", "Cabang Cirebon"),
            ("Unit", a_id, b_id, c_id, "Gudang Barat"),
            ("Kategori", "Kemeja Katun", "Jaket Parka", "Sandal Kulit", "Gerai Lampung"),
            ("Unit", a_id, b_id, c_id, "Gudang Timur"),
            ("Menu", "Kopi Tubruk", "Brownies Cokelat", "Matcha Latte", "Cabang Pontianak"),
            ("Unit", a_id, b_id, c_id, "Gudang Selatan"),
        ]
        months_id = ("Januari", "Februari", "Maret", "April", "Mei", "Juni")
        month_tag = months_id[(i // 6) % len(months_id)]
        noun_cmp, e1, e2, e3, loc_cmp = cmp_sets_id[i % len(cmp_sets_id)]
        p1, p2, p3 = 18 + (i * 5) % 25, 22 + (i * 7) % 29, 15 + (i * 11) % 33
        if p2 == p1:
            p2 += 3
        while p3 in (p1, p2):
            p3 += 2
        pairs_id = [(e1, p1), (e2, p2), (e3, p3)]
        if i % 2 == 0:
            win_ent, win_p = max(pairs_id, key=lambda x: x[1])
            lose_ent, lose_p = min(pairs_id, key=lambda x: x[1])
            sup = ("terbesar" if (i // 2) % 2 == 0 else "tertinggi")
            q_cmp = f"{noun_cmp} mana yang menyumbang kontribusi omzet {sup} di {loc_cmp} pada bulan {month_tag}?"
            a_cmp = f"Berdasarkan ringkasan {loc_cmp} [1], kontribusi omzet {sup} berasal dari {win_ent} sebesar {win_p}%."
            d_cmp = f"Berdasarkan ringkasan {loc_cmp} [1], kontribusi omzet {sup} berasal dari {lose_ent} sebesar {lose_p}%."
            ev_unit = "omzet"
        else:
            win_ent, win_p = min(pairs_id, key=lambda x: x[1])
            lose_ent, lose_p = max(pairs_id, key=lambda x: x[1])
            sup = ("terendah" if (i // 2) % 2 == 0 else "terkecil")
            q_cmp = f"{noun_cmp} mana yang mencatat tingkat ketepatan {sup} di {loc_cmp} pada bulan {month_tag}?"
            a_cmp = f"Berdasarkan audit {loc_cmp} [1], tingkat ketepatan {sup} tercatat pada {win_ent} sebesar {win_p}%."
            d_cmp = f"Berdasarkan audit {loc_cmp} [1], tingkat ketepatan {sup} tercatat pada {lose_ent} sebesar {lose_p}%."
            ev_unit = "ketepatan"
        target.append({
            "language": "id", "task": "grounded_compare_minmax", "business": True,
            "user": q_cmp,
            "evidence": f"[1] Ringkasan {loc_cmp} ({month_tag})\n{e1}: {p1}% {ev_unit}, {e2}: {p2}% {ev_unit}, {e3}: {p3}% {ev_unit}.",
            "assistant": a_cmp,
            "distractor": d_cmp,
        })
        # 3. ID unanswerable with strong distractor numbers (explicit money-vs-headcount/volume contrast!)
        unans_topics_id = [
            ("Berapa laba bersih cabang ini pada kuartal berjalan?", f"Total pengunjung toko: {vol} orang, dengan {v2 + 120} transaksi selesai.", "laba bersih", f"{vol} orang"),
            ("Berapa margin laba bersih gerai ini pada periode laporan?", f"Jumlah pengunjung gerai: {vol + 95} orang, dengan {v1 + 210} struk tercetak.", "margin laba bersih", f"{vol + 95} orang"),
            ("Berapa biaya listrik bulanan di fasilitas tersebut?", f"Jumlah karyawan tetap: {v2 + 10} orang dengan total {vol} pengiriman selesai.", "biaya listrik bulanan", f"{vol} pengiriman"),
            ("Mengapa tingkat keterlambatan kurir meningkat pekan lalu?", f"Total pesanan pekan lalu: {vol} transaksi, naik {v2} transaksi dibanding pekan sebelumnya.", "penyebab kenaikan keterlambatan kurir", f"{v2} transaksi"),
            ("Berapa persen potongan harga rata-rata untuk pelanggan grosir?", f"Jumlah mitra aktif: {v3 + 20} toko, dengan {v1} mitra baru bergabung bulan ini.", "persen potongan harga rata-rata", f"{v3 + 20} toko"),
        ]
        uq_id, uev_id, umiss_id, udist_id = unans_topics_id[i % len(unans_topics_id)]
        target.append({
            "language": "id", "task": "grounded_unans_distractor", "business": True,
            "user": f"{uq_id} ({a_id})",
            "evidence": f"[1] Laporan Kunjungan & Aktivitas {a_id}\n{uev_id}",
            "assistant": f"Kutipan [1] tidak menyebutkan {umiss_id}. Bisa bagikan data {umiss_id} yang terkait?",
            "distractor": f"Berdasarkan laporan {a_id} [1], {umiss_id} adalah: {udist_id}.",
        })

        # EN 3-part breakdown lookup (Total / Passed-On-Time / Failed-Overdue -> ask specifically for Failed!)
        tot_items = 800 + (i * 41) % 700
        failed_items = 18 + (i * 7) % 65
        passed_items = tot_items - failed_items
        b_domains = (
            ("deliveries", "delivered on time", "missed the delivery window"),
            ("invoices", "paid within terms", "became overdue"),
            ("batches", "passed quality inspection", "failed quality inspection"),
            ("shipments", "cleared customs on schedule", "were delayed at customs"),
        )
        noun, ok_verb, bad_verb = b_domains[i % len(b_domains)]
        target.append({
            "language": "en", "task": "grounded_breakdown_subset", "business": True,
            "user": f"How many {noun} {bad_verb} in cycle #{i + 1}?",
            "evidence": (
                f"[1] Cycle #{i + 1} Operations Digest\n"
                f"Total {noun}: {tot_items:,}. {ok_verb.capitalize()}: {passed_items:,}. "
                f"{bad_verb.capitalize()}: {failed_items} {noun}."
            ),
            "assistant": f"According to the Operations Digest [1], {failed_items} {noun} {bad_verb}.",
            "distractor": f"According to the Operations Digest [1], {passed_items:,} {noun} {bad_verb}.",
        })

        # EN counterparts (multi-entity lookup, min/max compare, distractor-number unanswerable)
        a_en, b_en, c_en = triples_en[i % len(triples_en)]
        u1 = round(1.10 + ((i * 7) % 35) * 0.15, 2)
        u2 = round(2.25 + ((i * 11) % 29) * 0.15, 2)
        u3 = round(0.85 + ((i * 13) % 23) * 0.15, 2)
        if u2 == u1:
            u2 = round(u2 + 0.45, 2)
        if u3 in (u1, u2):
            u3 = round(u3 + 0.65, 2)
        chosen_en, chosen_cost = (b_en, u2) if (i % 2 == 0) else (c_en, u3)
        target.append({
            "language": "en", "task": "grounded_multi_entity", "business": True,
            "user": f"What was the unit handling cost reported for {chosen_en} in period {i + 1}?",
            "evidence": (
                f"[1] Operations Unit Cost Digest #{i + 1}\n"
                f"Total orders processed: {vol:,}. {a_en}: ${u1:.2f}/order. {b_en}: ${u2:.2f}/order. {c_en}: ${u3:.2f}/order."
            ),
            "assistant": f"According to the Operations Unit Cost Digest [1], unit handling cost for {chosen_en} was ${chosen_cost:.2f}/order.",
            "distractor": f"Based on the provided excerpt [1], the answer is ${u1:.2f}.",
        })
        acc1 = round(92.0 + ((i * 3) % 70) * 0.1, 1)
        acc2 = round(91.5 + ((i * 7) % 75) * 0.1, 1)
        acc3 = round(93.0 + ((i * 11) % 60) * 0.1, 1)
        if acc2 == acc1:
            acc2 = round(acc2 + 0.4, 1)
        while acc3 in (acc1, acc2):
            acc3 = round(acc3 - 0.3, 1)
        pairs_en = [(a_en, acc1), (b_en, acc2), (c_en, acc3)]
        if i % 2 == 0:
            win_en, win_acc = min(pairs_en, key=lambda x: x[1])
            lose_en, lose_acc = max(pairs_en, key=lambda x: x[1])
            q_en_cmp = f"Which unit had the lowest fulfillment accuracy in audit #{i + 1}?"
            a_en_cmp = f"According to the audit [1], {win_en} had the lowest fulfillment accuracy at {win_acc}%."
            d_en_cmp = f"Based on the provided excerpt [1], {lose_en} had the lowest accuracy at {lose_acc}%."
        else:
            win_en, win_acc = max(pairs_en, key=lambda x: x[1])
            lose_en, lose_acc = min(pairs_en, key=lambda x: x[1])
            q_en_cmp = f"Which unit achieved the highest fulfillment accuracy in audit #{i + 1}?"
            a_en_cmp = f"According to the audit [1], {win_en} achieved the highest fulfillment accuracy at {win_acc}%."
            d_en_cmp = f"Based on the provided excerpt [1], {lose_en} achieved the highest accuracy at {lose_acc}%."
        target.append({
            "language": "en", "task": "grounded_compare_minmax", "business": True,
            "user": q_en_cmp,
            "evidence": f"[1] Weekly Fulfillment Audit #{i + 1}\n{a_en}: {acc1}% accuracy. {b_en}: {acc2}% accuracy. {c_en}: {acc3}% accuracy.",
            "assistant": a_en_cmp,
            "distractor": d_en_cmp,
        })
        unans_topics_en = [
            ("What was the monthly facility rent for that hub?", f"Active staff count: {v2 + 12} people across {vol:,} completed shipments.", "monthly facility rent", f"{vol:,} shipments"),
            ("Why did customer cancellations spike last Tuesday?", f"Total inbound bookings were {vol:,}, down {v2} from the prior week.", "the cause of cancellations", f"{v2}"),
            ("What was the average partner referral fee per signup?", f"Partner channel generated {vol:,} sessions and {v3 + 15} trial signups.", "the referral fee per signup", f"{v3 + 15}"),
        ]
        uq_en, uev_en, umiss_en, udist_en = unans_topics_en[i % len(unans_topics_en)]
        target.append({
            "language": "en", "task": "grounded_unans_distractor", "business": True,
            "user": f"{uq_en} ({a_en})",
            "evidence": f"[1] {a_en} Activity Brief\n{uev_en}",
            "assistant": f"The provided excerpt [1] does not specify {umiss_en}. Could you share {umiss_en}?",
            "distractor": f"Based on the provided excerpt [1], the answer is {udist_en}.",
        })

    # 4. Numeric-context operational decision questions with unknown drivers
    # (teaches asking ONE focused clarifying question for both missing drivers, NOT calling TOOL: kpi!)
    decision_train = [
        ("id", "Pendapatan cabang turun dari Rp80 juta ke Rp64 juta bulan ini, tetapi kami belum tahu apakah jumlah transaksi atau rata-rata keranjang yang turun. Haruskah kami langsung memberi promo besar?",
         "Berapa jumlah transaksi dan rata-rata nilai keranjang pada masing-masing bulan tersebut?"),
        ("id", "Penjualan toko online turun dari Rp45 juta ke Rp36 juta pekan ini, sementara kami belum cek apakah kunjungan atau tingkat checkout yang turun. Apakah kami harus langsung menaikkan anggaran iklan?",
         "Berapa jumlah kunjungan dan tingkat checkout pada kedua pekan tersebut?"),
        ("id", "Omzet katering turun dari Rp120 juta ke Rp95 juta kuartal ini, dan kami belum memeriksa apakah jumlah klien atau frekuensi pesan yang berkurang. Sebaiknya kami langsung memangkas menu?",
         "Berapa jumlah klien aktif dan frekuensi pesanan pada masing-masing kuartal tersebut?"),
        ("id", "Pendapatan gerai kopi turun dari Rp35 juta ke Rp28 juta minggu ini, tetapi kami belum tahu apakah jumlah pembeli harian atau nilai belanja per struk yang turun. Haruskah jam buka langsung diperpanjang?",
         "Berapa jumlah pembeli harian dan rata-rata nilai belanja per struk pada kedua minggu tersebut?"),
        ("id", "Omzet penjualan grosir turun dari Rp150 juta ke Rp118 juta bulan lalu, namun kami belum mengecek apakah jumlah pesanan atau ukuran order rata-rata yang turun. Apakah kami harus langsung menurunkan harga jual?",
         "Berapa jumlah pesanan dan rata-rata ukuran order pada masing-masing bulan tersebut?"),
        ("id", "Penjualan aplikasi turun dari Rp90 juta ke Rp74 juta, tapi tim belum tahu apakah jumlah pengguna aktif atau rasio transaksi yang menurun. Perlukah kami langsung merilis voucher gratis ongkir?",
         "Berapa jumlah pengguna aktif dan jumlah transaksi selesai pada kedua periode tersebut?"),
        ("en", "Weekly store sales dropped from $36,000 to $29,000, and we have not checked whether foot traffic or average basket size fell. Should we run a sitewide discount tomorrow?",
         "What were your foot traffic and average basket size in both weeks?"),
        ("en", "Online revenue slipped from $82,000 to $68,000 this month, yet we do not know whether sessions or order conversion declined. Should we increase ad spend right now?",
         "What were your total sessions and order conversion rate in each month?"),
        ("en", "Subscription billing fell from $44,000 to $37,000 in Q2, and we haven't checked whether new signups or renewal rate dropped. Should we cut plan prices immediately?",
         "What were your new signups and renewal rate across both quarters?"),
        ("en", "Wholesale sales fell from $110,000 to $88,000 last month, and we haven't checked whether active accounts or order frequency dropped. Should we double sales commissions today?",
         "What were your active account count and order frequency for each month?"),
        ("en", "Marketplace revenue dropped from $28,000 to $21,500 this week, and we don't know if product page views or checkout completion fell. Should we launch a flash sale today?",
         "What were your product page views and checkout completion counts in both weeks?"),
        ("en", "Branch sales declined from $65,000 to $54,000 this month, but we have not verified whether buyer count or average ticket dropped. Should we slash prices across all items?",
         "What were your buyer count and average ticket size in both months?"),
    ]
    decision_val = [
        ("id", "Pendapatan toko turun dari Rp72 juta ke Rp58 juta bulan ini, dan kami belum tahu apakah jumlah pembeli atau nilai belanja rata-rata yang berkurang. Haruskah kami langsung memasang diskon besar?",
         "Berapa jumlah pembeli dan rata-rata nilai belanja pada kedua bulan tersebut?"),
        ("en", "Monthly sales slipped from $74,000 to $61,000, and we haven't checked whether store visits or checkout orders fell. Should we double our marketing budget today?",
         "What were your store visits and completed checkout orders in both months?"),
    ]
    for lang, u, a in decision_train:
        train_rows.append({"language": lang, "task": "decision_missing_drivers", "business": True, "user": u, "assistant": a})
    for lang, u, a in decision_val:
        val_rows.append({"language": lang, "task": "decision_missing_drivers", "business": True, "user": u, "assistant": a})

    return train_rows, val_rows


def build_aya_indonesian_rows(limit: int = 750) -> tuple[list[dict], list[dict]]:
    """Human-curated Indonesian instructions from CohereLabs/aya_dataset (Apache-2.0)."""
    url = _resolve_url("https://huggingface.co/api/datasets/CohereLabs/aya_dataset/parquet/default/train/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for batch in pf.iter_batches(batch_size=4096, columns=["inputs", "targets", "language_code"]):
        for u, a, code in zip(
            batch.column("inputs").to_pylist(),
            batch.column("targets").to_pylist(),
            batch.column("language_code").to_pylist(),
        ):
            if code != "ind":
                continue
            u_s, a_s = (u or "").strip(), (a or "").strip()
            if not u_s or not a_s or len(a_s) < 20 or len(u_s) + len(a_s) > 1800:
                continue
            row = {"language": "id", "task": "aya_id", "business": False, "user": u_s, "assistant": a_s}
            if count % 15 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)
            count += 1
            if count >= limit:
                return train_rows, val_rows
    return train_rows, val_rows


def build_tydiqa_indonesian_rows(ans_limit: int = 400, unans_limit: int = 160) -> tuple[list[dict], list[dict]]:
    """Indonesian context-grounded QA + natural abstention from google-research-datasets/tydiqa (Apache-2.0)."""
    url = _resolve_url("https://huggingface.co/api/datasets/google-research-datasets/tydiqa/parquet/primary_task/train/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    n_ans, n_unans = 0, 0
    for batch in pf.iter_batches(
        batch_size=512,
        columns=["language", "question_text", "document_title", "document_plaintext", "passage_answer_candidates", "annotations"],
    ):
        for lang, q, title, doc, cands, ann in zip(
            batch.column("language").to_pylist(),
            batch.column("question_text").to_pylist(),
            batch.column("document_title").to_pylist(),
            batch.column("document_plaintext").to_pylist(),
            batch.column("passage_answer_candidates").to_pylist(),
            batch.column("annotations").to_pylist(),
        ):
            if lang != "indonesian":
                continue
            q_s = (q or "").strip()
            if not q_s or not doc:
                continue
            doc_b = doc.encode("utf-8")
            pidx = ann["passage_answer_candidate_index"][0]
            starts = cands["plaintext_start_byte"]
            ends = cands["plaintext_end_byte"]
            if pidx >= 0 and pidx < len(starts) and n_ans < ans_limit:
                passage = doc_b[starts[pidx]:ends[pidx]].decode("utf-8", errors="ignore").strip()
                ms = ann["minimal_answers_start_byte"][0]
                me = ann["minimal_answers_end_byte"][0]
                minimal = doc_b[ms:me].decode("utf-8", errors="ignore").strip() if ms >= 0 and me > ms else ""
                if not passage or len(passage) > 1000 or not minimal or len(minimal) > 250:
                    continue
                row = {
                    "language": "id", "task": "tydiqa_id_answerable", "business": True,
                    "user": q_s,
                    "evidence": f"[1] {title}\n{passage}",
                    "assistant": f"Berdasarkan kutipan {title} [1], jawabannya adalah: {minimal}.",
                }
                if n_ans % 14 == 0:
                    val_rows.append(row)
                else:
                    train_rows.append(row)
                n_ans += 1
            elif pidx < 0 and starts and n_unans < unans_limit:
                # Pick first clean short paragraph from the document as insufficient context
                passage = doc_b[starts[0]:ends[0]].decode("utf-8", errors="ignore").strip()
                if not passage or len(passage) < 80 or len(passage) > 900:
                    continue
                row = {
                    "language": "id", "task": "tydiqa_id_unanswerable", "business": True,
                    "user": q_s,
                    "evidence": f"[1] {title}\n{passage}",
                    "assistant": f"Kutipan mengenai {title} [1] tidak menyebutkan jawaban untuk pertanyaan tersebut. Bisa sertakan bagian atau informasi tambahan yang memuat rincian itu?",
                }
                if n_unans % 14 == 0:
                    val_rows.append(row)
                else:
                    train_rows.append(row)
                n_unans += 1
            if n_ans >= ans_limit and n_unans >= unans_limit:
                return train_rows, val_rows
    return train_rows, val_rows


def build_bitext_support_rows(limit: int = 150) -> tuple[list[dict], list[dict]]:
    """Customer support & order operations dialogues from Bitext (CDLA-Sharing-1.0).

    Capped at 150 rows and marked `business=False` with call-center boilerplate filtered out
    so canned phrases like "please don't hesitate" never leak into business consulting mode.
    """
    url = _resolve_url("https://huggingface.co/api/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/parquet/default/train/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    seen_intents: Counter[str] = Counter()
    count = 0
    for batch in pf.iter_batches(batch_size=512, columns=["instruction", "intent", "response"]):
        for u, intent, a in zip(
            batch.column("instruction").to_pylist(),
            batch.column("intent").to_pylist(),
            batch.column("response").to_pylist(),
        ):
            if seen_intents[intent] >= 10:
                continue
            u_s = _fill_placeholders((u or "").strip())
            a_s = _fill_placeholders((a or "").strip())
            low_a = a_s.casefold()
            if (
                "{{" in u_s or "{{" in a_s or not u_s or not a_s
                or len(u_s) + len(a_s) > 1500
                or "don't hesitate" in low_a or "further assistance" in low_a
            ):
                continue
            seen_intents[intent] += 1
            row = {"language": "en", "task": "bitext_support", "business": False, "user": u_s, "assistant": a_s}
            if count % 14 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)
            count += 1
            if count >= limit:
                return train_rows, val_rows
    return train_rows, val_rows


def build_sales_consultant_rows(limit: int = 400) -> tuple[list[dict], list[dict]]:
    """High-rated sales & retail consulting turns from Dauji-AI/SalesAgent-Consultant-V_0.0.1 (Apache-2.0)."""
    url = _resolve_url("https://huggingface.co/api/datasets/Dauji-AI/SalesAgent-Consultant-V_0.0.1/parquet/default/train/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for batch in pf.iter_batches(batch_size=512, columns=["conversations", "industry", "score"]):
        for conv, ind, score in zip(
            batch.column("conversations").to_pylist(),
            batch.column("industry").to_pylist(),
            batch.column("score").to_pylist(),
        ):
            try:
                if score is not None and float(score) < 4.5:
                    continue
            except (TypeError, ValueError):
                pass
            if not isinstance(conv, list) or len(conv) < 2:
                continue
            u = str(conv[0].get("value") or "").strip()
            a = str(conv[1].get("value") or "").strip()
            if not u or not a or len(u) < 25 or len(a) < 50 or len(u) + len(a) > 1500:
                continue
            prefix = f"[Industry: {ind}] " if ind else ""
            row = {"language": "en", "task": "sales_consulting", "business": False, "user": f"{prefix}{u}", "assistant": a}
            if count % 14 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)
            count += 1
            if count >= limit:
                return train_rows, val_rows
    return train_rows, val_rows


def build_squad_v2_rows(ans_limit: int = 350, unans_limit: int = 180) -> tuple[list[dict], list[dict]]:
    """Balanced context-grounded answerable & unanswerable QA from rajpurkar/squad_v2 (CC-BY-SA-4.0)."""
    url = _resolve_url("https://huggingface.co/api/datasets/rajpurkar/squad_v2/parquet/squad_v2/train/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    n_ans, n_unans = 0, 0
    seen_titles: Counter[str] = Counter()
    for batch in pf.iter_batches(batch_size=512, columns=["title", "context", "question", "answers"]):
        for title, ctx, q, ans in zip(
            batch.column("title").to_pylist(),
            batch.column("context").to_pylist(),
            batch.column("question").to_pylist(),
            batch.column("answers").to_pylist(),
        ):
            if seen_titles[title] >= 4:
                continue
            ctx_s, q_s = (ctx or "").strip(), (q or "").strip()
            if not ctx_s or not q_s or len(ctx_s) > 1000:
                continue
            texts = ans.get("text", []) if isinstance(ans, dict) else []
            if texts and n_ans < ans_limit:
                seen_titles[title] += 1
                row = {
                    "language": "en", "task": "squad_answerable", "business": True,
                    "user": q_s,
                    "evidence": f"[1] {title.replace('_', ' ')}\n{ctx_s}",
                    "assistant": f"Based on the provided excerpt [1], the answer is {texts[0].strip()}.",
                }
                if n_ans % 14 == 0:
                    val_rows.append(row)
                else:
                    train_rows.append(row)
                n_ans += 1
            elif not texts and n_unans < unans_limit:
                seen_titles[title] += 1
                row = {
                    "language": "en", "task": "squad_unanswerable", "business": True,
                    "user": q_s,
                    "evidence": f"[1] {title.replace('_', ' ')}\n{ctx_s}",
                    "assistant": "The provided excerpt [1] does not contain enough information to answer that question. Could you provide the missing details or source document?",
                }
                if n_unans % 14 == 0:
                    val_rows.append(row)
                else:
                    train_rows.append(row)
                n_unans += 1
            if n_ans >= ans_limit and n_unans >= unans_limit:
                return train_rows, val_rows
    return train_rows, val_rows


def build_ambig_qa_clarification_rows(limit: int = 350) -> tuple[list[dict], list[dict]]:
    """Explicit clarification-question training from sewon/ambig_qa (CC-BY-SA-3.0)."""
    url = _resolve_url("https://huggingface.co/api/datasets/sewon/ambig_qa/parquet/light/train/0.parquet")
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for batch in pf.iter_batches(batch_size=512, columns=["question", "annotations"]):
        for q, ann in zip(batch.column("question").to_pylist(), batch.column("annotations").to_pylist()):
            if not isinstance(ann, dict):
                continue
            types = ann.get("type") or []
            qapairs = ann.get("qaPairs") or []
            if "multipleQAs" not in types:
                continue
            idx = types.index("multipleQAs")
            sub_qs = qapairs[idx].get("question", []) if idx < len(qapairs) and isinstance(qapairs[idx], dict) else []
            clean_qs = [s.strip() for s in sub_qs if isinstance(s, str) and len(s.strip()) > 12][:2]
            if len(clean_qs) < 2:
                continue
            row = {
                "language": "en", "task": "ambig_clarification", "business": False,
                "user": q.strip(),
                "assistant": f"That question could mean more than one thing. To give an accurate answer, could you clarify whether you mean: (1) {clean_qs[0]} or (2) {clean_qs[1]}?",
            }
            if count % 14 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)
            count += 1
            if count >= limit:
                return train_rows, val_rows
    return train_rows, val_rows


def build_finqa_deliverable_rows(limit: int = 450) -> tuple[list[dict], list[dict]]:
    """Structured financial table + calculation deliverables from bevaya/FinQA (MIT)."""
    url = "https://huggingface.co/datasets/bevaya/FinQA/resolve/main/data/train-00000-of-00001.parquet"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as r:
        table = pq.read_table(io.BytesIO(r.read()))

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for pre, tbl, q, prog, ans, gold in zip(
        table["pre_text"].to_pylist(), table["table_ori"].to_pylist(),
        table["question"].to_pylist(), table["program"].to_pylist(),
        table["exe_ans"].to_pylist(), table["gold_inds"].to_pylist(),
    ):
        if count >= limit:
            break
        if not tbl or not q or not ans:
            continue
        rows_md = []
        for r_idx, row in enumerate(tbl[:8]):
            cells = [str(c).strip() for c in row]
            rows_md.append("| " + " | ".join(cells) + " |")
            if r_idx == 0 and cells:
                rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
        tbl_md = "\n".join(rows_md)
        evidence_lines = []
        if isinstance(gold, str) and gold.strip().startswith("{"):
            try:
                evidence_lines = [str(v).strip() for v in json.loads(gold).values() if v]
            except Exception:
                pass
        if not evidence_lines and pre:
            evidence_lines = [str(x).strip() for x in pre[:2] if x and str(x).strip() != "."]

        user_prompt = (
            f"Analyze the financial report data below and answer: {q}\n\n"
            f"Reference Table:\n{tbl_md}\n"
            + ("\nNotes:\n" + "\n".join(f"- {l}" for l in evidence_lines) if evidence_lines else "")
        )
        assistant_resp = (
            "### Financial Analysis Summary\n\n"
            + ("**Extracted Figures:**\n" + "\n".join(f"- {l}" for l in evidence_lines) + "\n\n" if evidence_lines else "")
            + f"**Calculation:** `{prog}`\n"
            + f"**Verified Result:** **{ans}**"
        )
        if len(user_prompt) + len(assistant_resp) > 2200:
            continue
        row = {"language": "en", "task": "finqa_deliverable", "business": False, "user": user_prompt, "assistant": assistant_resp}
        if count % 14 == 0:
            val_rows.append(row)
        else:
            train_rows.append(row)
        count += 1
    return train_rows, val_rows


def build_tatqa_rows(limit: int = 450) -> tuple[list[dict], list[dict]]:
    """Hybrid financial table + text QA from next-tat/TAT-QA (CC-BY-4.0)."""
    url = "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_train.json"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for item in data:
        if count >= limit:
            break
        tbl = (item.get("table") or {}).get("table") or []
        paras = [p.get("text", "").strip() for p in (item.get("paragraphs") or []) if p.get("text")]
        if not tbl:
            continue
        rows_md = []
        for r_idx, r_cells in enumerate(tbl[:8]):
            cells = [str(c).strip() for c in r_cells]
            rows_md.append("| " + " | ".join(cells) + " |")
            if r_idx == 0 and cells:
                rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
        tbl_md = "\n".join(rows_md)
        ctx = "\n".join(paras[:2])[:600]
        for q_obj in (item.get("questions") or [])[:2]:
            if count >= limit:
                break
            q_s = (q_obj.get("question") or "").strip()
            ans = q_obj.get("answer")
            deriv = (q_obj.get("derivation") or "").strip()
            scale = (q_obj.get("scale") or "").strip()
            if not q_s or ans is None or ans == "":
                continue
            ans_str = ", ".join(str(x) for x in ans) if isinstance(ans, list) else str(ans)
            if scale:
                ans_str = f"{ans_str} ({scale})"
            user_prompt = f"Financial Statement Excerpt:\n{tbl_md}\n\nNotes: {ctx}\n\nQuestion: {q_s}"
            assistant_resp = (
                (f"Calculation: `{deriv}`\n" if deriv else "")
                + f"Based on the financial statement, the answer is **{ans_str}**."
            )
            if len(user_prompt) + len(assistant_resp) > 2100:
                continue
            row = {"language": "en", "task": "tatqa_finance", "business": False, "user": user_prompt, "assistant": assistant_resp}
            if count % 14 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)
            count += 1
    return train_rows, val_rows


def build_balanced_calibration_rows() -> tuple[list[dict], list[dict]]:
    """Balanced, non-collapsing knowledge-boundary calibration rows (per Abstain-R1).

    Critically:
      - NO 720-row bare 'I do not know.' flood!
      - Ratio is ~1.5 : 1 (known : abstain) with varied phrasings so the model never
        collapses into a surface-trigger refusal shortcut.
    """
    known_facts = [
        ("Finance", "Accounting", "Under double-entry bookkeeping, what is the fundamental accounting equation?", "Assets = Liabilities + Equity."),
        ("Finance", "Corporate finance", "In CAPM, what Greek letter measures an asset's sensitivity to market returns?", "Beta."),
        ("Finance", "Macroeconomics", "What indicator measures the total value of finished goods and services produced within a country's borders?", "Gross Domestic Product (GDP)."),
        ("Finance", "Microeconomics", "At what condition does a profit-maximizing firm produce?", "Where Marginal Revenue equals Marginal Cost (MR = MC)."),
        ("Finance", "Fixed income", "How do fixed-rate bond prices move when market interest rates rise?", "Bond prices move inversely: when interest rates rise, bond prices fall."),
        ("Finance", "Accounting", "What is working capital?", "Working capital is Current Assets minus Current Liabilities."),
        ("Finance", "Corporate finance", "What does EBITDA stand for?", "Earnings Before Interest, Taxes, Depreciation, and Amortization."),
        ("Finance", "Corporate finance", "What does ROI stand for in business finance?", "Return on Investment."),
        ("Finance", "Operations", "In inventory management, what does EOQ stand for?", "Economic Order Quantity."),
        ("Finance", "Operations", "In retail operations, what does SKU stand for?", "Stock Keeping Unit."),
        ("Finance", "Marketing", "What are the traditional Four Ps of the marketing mix?", "Product, Price, Place, and Promotion."),
        ("Finance", "Auditing", "What is gross profit margin?", "Gross profit divided by revenue, expressed as a percentage: (Revenue - COGS) / Revenue."),
        ("Finance", "Accounting", "Which financial statement summarizes revenues and expenses over a reporting period?", "The income statement (profit and loss statement)."),
        ("Finance", "Accounting", "Which financial statement reports assets, liabilities, and equity at a specific point in time?", "The balance sheet."),
        ("Finance", "Accounting", "Under accrual accounting, when is revenue recognized?", "When it is earned (goods delivered or services performed), not necessarily when cash is received."),
        ("Law", "Contract Law", "What three elements form a binding common-law contract?", "Offer, acceptance, and consideration."),
        ("Law", "Corporations Law", "What legal rule protects directors for good-faith, informed business decisions?", "The business judgment rule."),
        ("Science Engineering and Mathematics", "Statistics", "What theorem states that sample means approach a normal distribution as sample size grows?", "The Central Limit Theorem."),
        ("Science Engineering and Mathematics", "Chemistry", "What is the pH of pure neutral water at 25 degrees Celsius?", "7."),
        ("Science Engineering and Mathematics", "Physics", "According to Newton's second law, how is net force related to mass and acceleration?", "F = ma (Force equals mass times acceleration)."),
        ("Health", "Physiology", "Which hormone secreted by the pancreas lowers blood glucose concentration?", "Insulin."),
        ("Software Engineering", "Python", "Which built-in Python data structure maps keys to values via a hash table?", "A dictionary (dict)."),
    ]

    unverifiable_templates = [
        ("Finance", "Accounting", "In the fiscal year {yr} annual report of {corp}, what exact dollar amount was recorded under {line_item}?"),
        ("Finance", "Auditing", "How many audit deficiencies were listed in Part I.A of PCAOB inspection report #{num} for {corp} in {yr}?"),
        ("Finance", "Mergers and acquisitions", "What exact EV/EBITDA multiple was disclosed in the {yr} private buyout proxy of {corp}?"),
        ("Finance", "Taxation", "Under Revenue Bulletin {yr}-{num}, what exact dollar threshold applied to equipment credit subsection {sec}?"),
        ("Law", "Administrative Law", "In {case_a} v. {case_b} ({yr}), what exact numerical threshold did the administrative panel adopt on remand?"),
        ("Science Engineering and Mathematics", "Chemistry", "In crystallographic report #{num} ({yr}), what exact bond angle to two decimal places was measured at the central atom?"),
        ("Health", "Epidemiology", "In regional cohort study #{num} ({yr}), what exact adjusted hazard ratio was reported for subgroup B?"),
        ("Humanities and Social Sciences", "Public policy", "In municipal circular #{num} ({yr}), what exact percentage of grant funds went to district 7?"),
    ]
    corps = ["Bavaria Textil AG", "Nordic Logistics SE", "Vanguard Industrial plc", "Helvetia Pharma SA", "Meridian Holdings Inc."]
    line_items = ["net cash flow hedge reserve", "deferred stripping asset amortization", "non-controlling translation adjustment"]
    cases_a = ["Harrington", "Kowalski", "Redwood Fabrication", "Westbrook Terminal"]
    cases_b = ["Metro Commission", "Apex Coating LLC", "Harbor View Board", "Summit Insurance"]

    abstain_answers = [
        "I do not know the exact figure without the source document.",
        "I do not know.",
        "I do not have access to that specific report or exact number, so I cannot answer reliably.",
    ]

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    # 2x phrasing variation for known facts (~44 rows)
    for rep in range(2):
        for idx, (dom, sub, q, ans) in enumerate(known_facts):
            prompt = (
                f"Domain: {dom} ({sub})\nQuestion: {q}\n"
                "Answer concisely if you know; otherwise state clearly that you do not know."
                if rep == 0 else q
            )
            row = {"language": "en", "task": "calibration_known_concept", "business": False, "user": prompt, "assistant": ans}
            if rep == 0 and idx % 6 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)

    # Only 64 unverifiable rows (NOT 720!) with varied abstain phrasings
    for i in range(64):
        dom, sub, tmpl = unverifiable_templates[i % len(unverifiable_templates)]
        q = tmpl.format(
            corp=corps[i % len(corps)], yr=1998 + (i * 3) % 26,
            line_item=line_items[i % len(line_items)], num=120 + (i * 19) % 850,
            sec=f"{1 + i % 8}.{i % 12}", case_a=cases_a[i % len(cases_a)], case_b=cases_b[(i + 1) % len(cases_b)],
        )
        prompt = (
            f"Domain: {dom} ({sub})\nQuestion: {q}\n"
            "Answer concisely if you know; otherwise state clearly that you do not know."
            if i % 2 == 0 else q
        )
        row = {
            "language": "en", "task": "calibration_unknown_abstain", "business": False,
            "user": prompt, "assistant": abstain_answers[i % len(abstain_answers)],
        }
        if i >= 56:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def build_smoltalk_rows(limit: int = 500) -> tuple[list[dict], list[dict]]:
    """General English instruction rows from HuggingFaceTB/smol-smoltalk (Apache-2.0)."""
    url = "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00000-of-00004.parquet"
    pf = pq.ParquetFile(HttpRangeFile(url))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for batch in pf.iter_batches(batch_size=256, columns=["messages"]):
        for msgs in batch.column(0).to_pylist():
            if count >= limit:
                return train_rows, val_rows
            if not msgs or len(msgs) < 2:
                continue
            u = str(msgs[0].get("content") or "").strip()
            a = str(msgs[1].get("content") or "").strip()
            if not u or not a or len(u) + len(a) > 1600 or len(a) < 40:
                continue
            row = {"language": "en", "task": "smoltalk", "business": False, "user": u, "assistant": a}
            if count % 14 == 0:
                val_rows.append(row)
            else:
                train_rows.append(row)
            count += 1
    return train_rows, val_rows


def build_agentic_tool_rows() -> tuple[list[dict], list[dict]]:
    """Build bilingual (EN + ID) Business Agent tool-calling & deliverable synthesis rows.

    Teaches three core Kuon Business Agent behaviors (100% disjoint from eval_business_agent.py):
      1. Emit `TOOL: kpi(...)` or `TOOL: calc(...)` when exact arithmetic is needed.
      2. Synthesize a concise grounded deliverable citing `[1]` verified tool/table output
         with 1 testable action + 1 measurable metric.
      3. Refuse to fabricate inputs and ask for missing numbers when required fields are absent.
    """
    from kuon_agent.tools import business_kpi

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    kpi_cases = [
        # (lang, metric, kwargs, user_q_tmpl, synth_action_metric)
        (
            "id", "gross_margin", {"revenue": 120000000, "cogs": 78000000},
            "Omzet bulan ini Rp120.000.000 dan HPP Rp78.000.000. Berapa gross margin kami dan apa langkah evaluasinya?",
            "Fakta terkonfirmasi [1]: gross margin sebesar {margin_pct}% (laba kotor Rp{gross_profit:,.0f}). "
            "Dugaan awal: biaya bahan pada beberapa produk mungkin menekan margin. "
            "Tindakan teruji: audit HPP 5 produk terlaris minggu ini. "
            "Metrik keberhasilan: pantau perubahan gross margin per produk setiap minggu.",
        ),
        (
            "en", "gross_margin", {"revenue": 95000, "cogs": 60800},
            "Monthly revenue was $95,000 and COGS was $60,800. What is our gross margin and what should we check next?",
            "Confirmed fact [1]: gross margin is {margin_pct}% (gross profit ${gross_profit:,.0f}). "
            "Hypothesis: supplier cost drift or discounting on top SKUs may be compressing margin. "
            "Testable action: audit unit COGS on the top 5 selling items this week. "
            "Metric: track weekly gross margin percentage by category.",
        ),
        (
            "id", "pct_change", {"old": 420, "new": 336},
            "Pesanan mingguan turun dari 420 menjadi 336 pesanan. Berapa persen penurunannya dan apa langkah pertama?",
            "Fakta terkonfirmasi [1]: pesanan turun {pct_change}% (selisih {delta:+.0f} pesanan). "
            "Dugaan: penurunan bisa berasal dari trafik pengunjung atau konversi pembelian. "
            "Tindakan teruji: pisahkan data kunjungan harian dan jumlah transaksi 14 hari terakhir. "
            "Metrik keberhasilan: identifikasi rasio konversi harian sebelum mengubah harga atau promosi.",
        ),
        (
            "en", "pct_change", {"old": 800, "new": 640},
            "Weekly orders dropped from 800 to 640. What is the exact percentage drop and first diagnostic step?",
            "Confirmed fact [1]: orders changed by {pct_change}% ({delta:+.0f} orders). "
            "Hypothesis: the drop is driven by either lower traffic or lower checkout conversion. "
            "Testable action: split daily visitors vs completed orders for the last 14 days. "
            "Metric: track daily visit-to-order conversion rate before changing prices.",
        ),
        (
            "id", "conversion_rate", {"buyers": 54, "visitors": 450},
            "Minggu ini ada 450 pengunjung toko dan 54 yang jadi membeli. Berapa conversion rate kami?",
            "Fakta terkonfirmasi [1]: conversion rate sebesar {rate_pct}%. "
            "Dugaan: sebagian pengunjung mungkin batal beli karena stok, antrean, atau informasi produk kurang jelas. "
            "Tindakan teruji: catat alasan utama pengunjung keluar tanpa membeli selama 3 hari jam ramai. "
            "Metrik keberhasilan: ukur kenaikan conversion rate harian pada jam ramai.",
        ),
        (
            "en", "conversion_rate", {"buyers": 63, "visitors": 525},
            "We had 525 store visitors and 63 buyers this week. What is our conversion rate?",
            "Confirmed fact [1]: conversion rate is {rate_pct}%. "
            "Hypothesis: visitors may be leaving due to stockouts, queue wait time, or unclear pricing. "
            "Testable action: log the top exit reason from 20 non-buying visitors over 3 peak shifts. "
            "Metric: monitor daily conversion rate after fixing the top blocker.",
        ),
        (
            "id", "breakeven_units", {"fixed_cost": 15000000, "unit_price": 75000, "unit_cost": 45000},
            "Biaya tetap Rp15.000.000, harga jual Rp75.000, dan biaya variabel Rp45.000 per unit. Berapa unit untuk balik modal?",
            "Fakta terkonfirmasi [1]: titik impas adalah {units} unit (margin kontribusi Rp{contribution_margin:,.0f}/unit). "
            "Tindakan teruji: buat target penjualan mingguan minimal seperempat titik impas dan cek selisihnya tiap Jumat. "
            "Metrik keberhasilan: jumlah unit terjual kumulatif dibanding target {units} unit per bulan.",
        ),
        (
            "en", "breakeven_units", {"fixed_cost": 12000, "unit_price": 80, "unit_cost": 50},
            "Fixed cost is $12,000, unit price is $80, and unit variable cost is $50. How many units to break even?",
            "Confirmed fact [1]: break-even volume is {units} units (contribution margin ${contribution_margin:,.0f}/unit). "
            "Testable action: set a weekly unit milestone of 25% of break-even and review variance every Friday. "
            "Metric: cumulative units sold against the {units}-unit monthly break-even threshold.",
        ),
        (
            "id", "aov", {"revenue": 48000000, "orders": 320},
            "Omzet minggu ini Rp48.000.000 dari 320 pesanan. Berapa nilai rata-rata per pesanan (AOV)?",
            "Fakta terkonfirmasi [1]: AOV sebesar Rp{aov:,.0f} per pesanan. "
            "Tindakan teruji: uji penawaran paket pendamping untuk 1 produk terlaris di kasir selama 7 hari. "
            "Metrik keberhasilan: bandingkan AOV minggu uji terhadap baseline Rp{aov:,.0f}.",
        ),
        (
            "en", "roi", {"gain": 18000, "cost": 12000},
            "A promo campaign generated $18,000 in gross contribution on $12,000 spend. What is the ROI?",
            "Confirmed fact [1]: campaign ROI is {roi_pct}% (net gain ${net_gain:,.0f}). "
            "Testable action: re-test the same creative on a narrow high-intent segment before scaling budget. "
            "Metric: track incremental net ROI and repeat purchase rate over 14 days.",
        ),
    ]

    # Expand deterministic numeric variations cleanly with INDEPENDENT per-argument scaling
    # so output ratios (margin%, ROI%, conversion%, pct_change%, break-even) vary genuinely
    # across train and val instead of collapsing to one constant percentage.
    num_scales = (1.00, 1.15, 0.85, 1.32, 0.74, 1.48, 1.22, 0.92, 1.65, 1.08, 1.37, 0.81)
    den_scales = (1.00, 1.04, 0.96, 1.18, 0.88, 1.25, 1.05, 1.12, 1.30, 0.91, 1.14, 0.97)
    for mult_idx in range(12):
        s_num = num_scales[mult_idx]
        s_den = den_scales[mult_idx]
        target = val_rows if mult_idx >= 10 else train_rows
        for lang, metric, base_kw, q_tmpl, ans_tmpl in kpi_cases:
            kw: dict[str, int] = {}
            for k_idx, (k, v) in enumerate(base_kw.items()):
                factor = s_num if k_idx == 0 else (s_den if k_idx == 1 else (0.5 * (s_num + s_den)))
                kw[k] = max(1, int(round(v * factor)))
            # Keep unit_price > unit_cost and margin/conversion sensible
            if "unit_price" in kw and kw["unit_price"] <= kw["unit_cost"]:
                kw["unit_price"] = kw["unit_cost"] + max(15, int(round(kw["unit_cost"] * 0.35)))
            if "buyers" in kw and "visitors" in kw and kw["buyers"] >= kw["visitors"]:
                kw["buyers"] = max(1, kw["visitors"] // 5)
            if "cogs" in kw and "revenue" in kw and kw["cogs"] >= kw["revenue"]:
                kw["cogs"] = max(1, int(round(kw["revenue"] * 0.68)))
            res = business_kpi(metric, **kw)
            arg_str = ", ".join([metric] + [f"{k}={v}" for k, v in kw.items()])
            tool_call = f"TOOL: kpi({arg_str})"
            tool_ev = f"[1] (tool:kpi) kpi({metric}): " + ", ".join(f"{k}={v}" for k, v in res.items() if k != "metric")

            # Format user question with the scaled numbers
            if lang == "id":
                q_text = f"Hitung metrik {metric} dari data klien berikut: " + ", ".join(f"{k}={v:,}" for k, v in kw.items()) + ". Apa rekomendasi langkah selanjutnya?"
            else:
                q_text = f"Calculate {metric} from these client numbers: " + ", ".join(f"{k}={v:,}" for k, v in kw.items()) + ". What should we test next?"
            if mult_idx == 0:
                q_text = q_tmpl

            # Turn 1: emit deterministic TOOL call
            target.append({
                "language": lang, "task": "agent_tool_call", "business": True,
                "user": q_text, "assistant": tool_call,
            })
            # Turn 2: synthesize grounded deliverable from verified [1] tool output
            target.append({
                "language": lang, "task": "agent_tool_synthesis", "business": True,
                "user": q_text, "evidence": tool_ev,
                "assistant": ans_tmpl.format(**res),
            })

    # Strictly disjoint train vs val missing-input abstention rows (zero train/val overlap!)
    train_missing_cases = [
        ("id", "Omzet toko kami Rp90 juta bulan ini. Berapa gross margin bersih kami?",
         "Data HPP (biaya pokok penjualan) belum tersedia, jadi gross margin tidak dapat dihitung dari angka omzet Rp90 juta saja. Berapa total HPP bulan ini?"),
        ("id", "Biaya tetap usaha kami Rp20 juta per bulan. Berapa unit yang harus dijual agar balik modal?",
         "Harga jual per unit dan biaya variabel per unit belum disebutkan, sehingga titik impas belum bisa dihitung. Berapa harga jual dan biaya produksi per unitnya?"),
        ("id", "Pengunjung toko online kami minggu ini 4.200 orang. Berapa conversion rate kami?",
         "Jumlah pembeli atau transaksi yang selesai belum tersedia, jadi conversion rate belum bisa dihitung. Berapa jumlah pembeli pada minggu yang sama?"),
        ("id", "Omzet minggu ini Rp32.000.000. Berapa rata-rata nilai pesanan (AOV) kami?",
         "Jumlah pesanan minggu ini belum disebutkan, sehingga AOV belum bisa dihitung. Berapa total pesanan yang menghasilkan omzet Rp32.000.000 tersebut?"),
        ("id", "Kami mengeluarkan biaya iklan Rp7.500.000 minggu lalu. Berapa ROI kampanye tersebut?",
         "Pendapatan atau kontribusi kotor dari iklan tersebut belum disebutkan, jadi ROI belum dapat dihitung. Berapa hasil penjualan yang berasal dari iklan itu?"),
        ("id", "Omzet cabang Bandung minggu ini Rp18.000.000. Berapa persen perubahan omzet dibanding minggu lalu?",
         "Omzet minggu sebelumnya belum disebutkan, sehingga persentase perubahan tidak dapat dihitung. Berapa omzet cabang Bandung minggu lalu?"),
        ("en", "Our revenue was $85,000 this month. What is our gross margin?",
         "Cost of goods sold (COGS) is missing, so gross margin cannot be calculated from revenue alone. What was your total COGS this month?"),
        ("en", "We spent $5,000 on ads this week. What was our campaign ROI?",
         "The revenue or gross contribution generated by the campaign is missing, so ROI cannot be calculated yet. How much attributable revenue or contribution came from those ads?"),
        ("en", "Our website had 9,400 visitors in May. What was our checkout conversion rate?",
         "The number of completed buyers or orders is missing, so conversion rate cannot be calculated yet. How many visitors completed a purchase in May?"),
        ("en", "Monthly fixed rent and payroll total $14,000. How many units must we sell to break even?",
         "Unit selling price and unit variable cost are missing, so break-even volume cannot be computed. What are your selling price and variable cost per unit?"),
        ("en", "Gross revenue reached $64,000 last week. What was our average order value?",
         "Total order count is missing, so average order value cannot be computed. How many completed orders generated that $64,000?"),
        ("en", "Week 2 store revenue was $22,400. By what percentage did revenue change from Week 1?",
         "Week 1 revenue is missing, so percentage change cannot be computed yet. What was the revenue figure for Week 1?"),
    ]
    val_missing_cases = [
        ("id", "Total HPP kedai kami Rp42.000.000 bulan ini. Berapa persen margin kotor kami?",
         "Total omzet bulan ini belum tersedia, sehingga margin kotor belum bisa dihitung dari HPP saja. Berapa total omzet pada bulan yang sama?"),
        ("id", "Tercatat 185 pembeli di aplikasi pekan ini. Berapa tingkat konversi toko kami?",
         "Jumlah total pengunjung aplikasi pada pekan yang sama belum tersedia, jadi tingkat konversi belum bisa dihitung. Berapa jumlah pengunjung pekan ini?"),
        ("en", "Our cost of goods sold was $38,500 in Q3. What was our gross margin percentage?",
         "Total Q3 revenue is missing, so gross margin cannot be calculated from COGS alone. What was your Q3 revenue?"),
        ("en", "We fulfilled 410 customer orders this week. What was our average order value?",
         "Total weekly revenue is missing, so average order value cannot be calculated from order count alone. What was total revenue across those 410 orders?"),
    ]
    for lang, u, a in train_missing_cases:
        train_rows.append({"language": lang, "task": "agent_tool_missing_input", "business": True, "user": u, "assistant": a})
    for lang, u, a in val_missing_cases:
        val_rows.append({"language": lang, "task": "agent_tool_missing_input", "business": True, "user": u, "assistant": a})

    return train_rows, val_rows


def build_rehearsal_rows(id_limit: int = 350, en_limit: int = 350) -> list[dict]:
    """Bilingual Experience Rehearsal (`raw_text: True`, ~15% of SFT) to prevent catastrophic forgetting.

    Streams clean Indonesian prose from epfml/FineWeb2-HQ (ODC-By 1.0) and business/finance
    textbook excerpts from crumb/openstax-text (CC BY 4.0).
    """
    rows: list[dict] = []
    # 1. Indonesian FineWeb2-HQ rehearsal (shard 2, unseen by Stage 1/2!)
    url_id = hf_shard_url("fineweb2_hq_id", 2)
    pf = pq.ParquetFile(HttpRangeFile(url_id))
    got_id = 0
    for batch in pf.iter_batches(batch_size=128, columns=["text"]):
        for raw in batch.column(0).to_pylist():
            t = clean_text(raw or "")
            if not t:
                continue
            rows.append({"language": "id", "task": "rehearsal_id", "raw_text": True, "text": t[:1400]})
            got_id += 1
            if got_id >= id_limit:
                break
        if got_id >= id_limit:
            break

    # 2. English OpenStax business/finance rehearsal
    got_en = 0
    for book in OPENSTAX_BUSINESS_FILES[:10]:
        if got_en >= en_limit:
            break
        url_en = f"https://huggingface.co/datasets/crumb/openstax-text/resolve/{OPENSTAX_REV}/{book}"
        try:
            with urlopen(Request(url_en, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-180000"}), timeout=25) as r:
                raw_txt = r.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        paras = [p.strip() for p in re.split(r"\n\n+", raw_txt) if len(p.strip()) >= 200]
        for p in paras[:40]:
            cleaned = clean_text(p)
            if not cleaned:
                continue
            rows.append({"language": "en", "task": "rehearsal_en", "raw_text": True, "text": cleaned[:1400]})
            got_en += 1
            if got_en >= en_limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/kuon1-bilingual-24k.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--offline", action="store_true", help="skip HF downloads (local sources only)")
    args = ap.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    biz_root = ROOT / "datasets/business"
    forbidden = set()
    for fname in ("eval.jsonl", "advice_eval.jsonl"):
        for line in (biz_root / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                forbidden.add(" ".join(json.loads(line)["user"].casefold().split()))
    from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES
    from scripts.eval_calibration import BUSINESS_CALIBRATION_CASES
    for c in CALIBRATION_CASES:
        forbidden.add(" ".join(c.user.casefold().split()))
    for c in DELIVERABLE_CASES:
        forbidden.add(" ".join(c.user.casefold().split()))
    for _, _, q, _, _ in BUSINESS_CALIBRATION_CASES:
        forbidden.add(" ".join(q.casefold().split()))

    train_all: list[dict] = []
    val_all: list[dict] = []

    def add_pair(label: str, fn):
        tr, va = fn()
        print(f"[kuon1-sft] loaded {label:24s}: {len(tr):4d} train | {len(va):3d} val", flush=True)
        train_all.extend(tr)
        val_all.extend(va)

    # Local high-precision business, tool-calling & calibration pillars (1x — NO 3x duplication!)
    add_pair("advice_pairs_1x", build_counterfactual_advice_rows)
    add_pair("grounded_business_1x", build_grounded_business_rows)
    add_pair("balanced_calibration", build_balanced_calibration_rows)
    add_pair("agentic_business_tools", build_agentic_tool_rows)

    if not args.offline:
        add_pair("aya_indonesian", build_aya_indonesian_rows)
        add_pair("tydiqa_indonesian", build_tydiqa_indonesian_rows)
        add_pair("bitext_support", build_bitext_support_rows)
        add_pair("sales_consultant", build_sales_consultant_rows)
        add_pair("squad_v2_balanced", build_squad_v2_rows)
        add_pair("ambig_qa_clarify", build_ambig_qa_clarification_rows)
        add_pair("finqa_deliverable", build_finqa_deliverable_rows)
        add_pair("tatqa_finance", build_tatqa_rows)
        add_pair("smol_smoltalk", build_smoltalk_rows)
        reh = build_rehearsal_rows()
        print(f"[kuon1-sft] loaded {'rehearsal_en_id':24s}: {len(reh):4d} train |   0 val", flush=True)
        train_all.extend(reh)

    def filter_fit(rows: list[dict]) -> list[dict]:
        kept = []
        for r in rows:
            if r.get("raw_text"):
                ids = tokenizer.encode(r["text"], add_eos=True)
                if 32 <= len(ids) <= args.block_size + 1:
                    kept.append(r)
                continue
            key = " ".join(r["user"].casefold().split())
            if key in forbidden:
                continue
            inp, lbl = build_example(
                tokenizer, r["user"], r["assistant"], args.block_size,
                evidence=r.get("evidence"), business=bool(r.get("business", False)),
            )
            if inp and lbl and lbl[-1] == tokenizer.eos_id:
                kept.append(r)
        return kept

    train_clean = filter_fit(train_all)
    val_clean = filter_fit(val_all)

    # Guarantee zero exact train/val prompt overlap across every task
    train_user_keys = {
        " ".join(r["user"].casefold().split())
        for r in train_clean if not r.get("raw_text") and r.get("user")
    }
    seen_val_keys: set[tuple[str, str]] = set()
    dedup_val: list[dict] = []
    for r in val_clean:
        if r.get("raw_text"):
            dedup_val.append(r)
            continue
        u_key = " ".join(r["user"].casefold().split())
        ev_key = " ".join((r.get("evidence") or "").casefold().split())
        if u_key in train_user_keys or (u_key, ev_key) in seen_val_keys:
            continue
        seen_val_keys.add((u_key, ev_key))
        dedup_val.append(r)
    val_clean = dedup_val

    rng = random.Random(1337)
    rng.shuffle(train_clean)
    rng.shuffle(val_clean)

    args.out.mkdir(parents=True, exist_ok=True)
    for split_name, rows in (("train", train_clean), ("val", val_clean)):
        path = args.out / f"sft_{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = Counter(r["task"] for r in rows)
        langs = Counter(r.get("language", "?") for r in rows)
        print(f"[kuon1-sft] {split_name}: {len(rows):,} rows | langs={dict(langs)} | tasks={dict(counts)}", flush=True)


if __name__ == "__main__":
    main()

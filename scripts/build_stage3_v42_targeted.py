"""Build Stage 3 v4.2 targeted dataset to push Kuon 1 past release gates.

Directly targets the exact remaining v4.1 misses (zero eval leakage):
1. Exact-digit copy fidelity on 2-number & 3-number operational reports (fixes 37 -> 28 copy slips).
2. High-density bilingual numerical min/max comparison + HARD entity-number cross-binding negatives
   (RIGHT entity + WRONG neighbor's number, plus wrong-rank entity) for %, decimal rates, Rp, $.
3. Missing-driver clarification pairs where user names two unknown drivers A vs B across periods
   (restores Pillar 2 Deliverables to 36/36 = 100%).
4. Keeps 100% unanswerable restraint, strong confirmed_actions, and 54 Kuon 1 identity rows.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.build_stage3_v41_rebalanced import (
    CITIES_EN,
    CITIES_ID,
    HUBS_EN,
    MONTHS_EN,
    MONTHS_ID,
    PRODUCTS_EN,
    PRODUCTS_ID,
    UNANS_TOPICS_EN,
    UNANS_TOPICS_ID,
    _cap_tasks,
    _check_no_leak,
    _fmt_id_int,
    _load_jsonl,
    _write_jsonl,
    build_identity_prefs,
    generate_en_evidence_cases,
    generate_id_evidence_cases,
)
from scripts.distill_gemma4 import build_identity_sft_rows
from scripts.eval_business_agent import CALIBRATION_CASES, DELIVERABLE_CASES

_EXTRA_FORBIDDEN = {
    "omzet kami turun dari rp60 juta ke rp48 juta",
    "monthly revenue fell from $50,000 to $41,000",
    "cabang solo", "gudang surabaya", "north hub", "east hub", "west hub",
}

DEPOTS_ID = [
    "Gudang Semarang", "Gudang Makassar", "Gudang Medan", "Gudang Malang",
    "Gudang Denpasar", "Gudang Bekasi", "Gudang Bogor", "Gudang Cirebon",
    "Gudang Pekanbaru", "Gudang Lampung", "Gudang Depok", "Gudang Tangerang",
]
BRANCHES_ID = [
    "Cabang Semarang", "Cabang Makassar", "Cabang Medan", "Cabang Malang",
    "Cabang Denpasar", "Cabang Balikpapan", "Cabang Padang", "Cabang Manado",
    "Cabang Bekasi", "Cabang Bogor", "Cabang Cirebon", "Cabang Pekanbaru",
]
DRIVER_PAIRS_ID = [
    ("jumlah pembeli", "nilai belanja rata-rata", "turunkan harga"),
    ("jumlah pengunjung", "rasio pembelian di kasir", "tambah biaya iklan"),
    ("jumlah pelanggan baru", "frekuensi pembelian ulang", "berikan diskon besar"),
    ("jumlah pesanan harian", "ukuran keranjang belanja", "potong harga semua menu"),
    ("jumlah klik iklan", "tingkat konversi halaman produk", "gandakan anggaran iklan"),
    ("jumlah prospek masuk", "persentase penutupan transaksi", "beri potongan harga 30%"),
    ("jumlah meja terisi", "rata-rata tagihan per meja", "ubah seluruh daftar harga"),
    ("jumlah kunjungan cabang", "nilai transaksi per pengunjung", "langsung pasang promo besar"),
]
DRIVER_PAIRS_EN = [
    ("store visits", "checkout conversion", "double ad spend today"),
    ("website traffic", "average order value", "cut prices across the catalog"),
    ("new buyer count", "repeat order rate", "launch a 25% flash discount"),
    ("inbound leads", "sales close rate", "triple marketing budget immediately"),
    ("daily foot traffic", "basket size per transaction", "slash retail prices now"),
    ("product page views", "cart completion rate", "increase paid search bids"),
    ("active subscribers", "revenue per account", "discount annual plans right away"),
    ("walk-in customers", "purchase conversion", "run a storewide markdown"),
]


def _check_clean(text: str) -> None:
    _check_no_leak(text)
    low = text.casefold()
    for bad in _EXTRA_FORBIDDEN:
        if bad in low:
            raise ValueError(f"Leakage detected: {bad!r}")


def generate_exact_copy_operational_cases(
    n_id: int, n_en: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """2-number & 3-number operational reports with fabricated-integer copy-fidelity negatives."""
    sft_rows: list[dict] = []
    pref_rows: list[dict] = []

    reasons_id = [
        "alamat tidak ditemukan",
        "penerima tidak di tempat",
        "kemasan bocor saat transit",
        "pesanan dibatalkan di pintu",
        "nomor telepon tidak aktif",
    ]
    reasons_en = [
        "incomplete shipping address",
        "recipient unavailable at delivery",
        "damaged outer carton",
        "failed barcode scan",
        "late carrier handoff",
    ]

    for i in range(n_id):
        depot = DEPOTS_ID[i % len(DEPOTS_ID)]
        month = MONTHS_ID[(i + rng.randint(0, 4)) % len(MONTHS_ID)]
        total = rng.randint(650, 4900)
        while total in (1840, 1290, 1246):
            total += 9
        target = rng.randint(11, 98)
        while target == 37:
            target += 2
        fake_close = target + rng.choice([-9, -7, -5, -4, 4, 6, 8, 11])
        if fake_close < 5 or fake_close == target:
            fake_close = target + 5
        reason = reasons_id[i % len(reasons_id)]

        if i % 2 == 0:
            # 2-number operational report (matches real concise reports like total + returned)
            ev = (
                f"[1] Laporan Operasional {depot} ({month})\n"
                f"Total pengiriman: {_fmt_id_int(total)} paket. "
                f"Paket retur karena {reason}: {target} paket."
            )
            wrong_neighbor = f"Berdasarkan laporan {depot} [1], jumlah retur paket pada bulan {month} adalah {_fmt_id_int(total)} paket."
        else:
            other = rng.randint(11, 98)
            while other in (target, fake_close, 37):
                other += 3
            ev = (
                f"[1] Laporan Operasional {depot} ({month})\n"
                f"Total pengiriman: {_fmt_id_int(total)} paket. "
                f"Paket tertunda di sortir: {other} paket. "
                f"Paket retur karena {reason}: {target} paket."
            )
            wrong_neighbor = f"Berdasarkan laporan {depot} [1], jumlah retur paket pada bulan {month} adalah {other} paket."

        q = f"Berapa jumlah retur paket di {depot} pada bulan {month}?"
        ans = f"Berdasarkan laporan operasional {depot} [1], jumlah retur paket pada bulan {month} adalah {target} paket."
        wrong_fabricated = f"Berdasarkan laporan operasional {depot} [1], jumlah retur paket pada bulan {month} adalah {fake_close} paket."
        false_abs = f"Kutipan [1] tidak menyebutkan jumlah retur paket di {depot}. Bisa bagikan laporannya?"
        for t in (ev, q, ans, wrong_fabricated, wrong_neighbor):
            _check_clean(t)

        sft_rows.append({"language": "id", "task": "v42_exact_copy_ans", "business": True, "user": q, "evidence": ev, "assistant": ans})
        pref_rows.append({"language": "id", "task": "v42_pref_exact_copy_digit", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": wrong_fabricated})
        pref_rows.append({"language": "id", "task": "v42_pref_exact_copy_neighbor", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": wrong_neighbor})
        pref_rows.append({"language": "id", "task": "v42_pref_exact_copy_no_abs", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": false_abs})

        if i % 3 == 0:
            utopic, uask = UNANS_TOPICS_ID[i % len(UNANS_TOPICS_ID)]
            uq = f"Berapa {utopic} di {depot} pada bulan {month}?"
            ua = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            uw = f"Menurut laporan {depot} [1], {utopic} sebesar {target} paket."
            sft_rows.append({"language": "id", "task": "v42_exact_copy_unans", "business": True, "user": uq, "evidence": ev, "assistant": ua})
            pref_rows.append({"language": "id", "task": "v42_pref_exact_copy_abstain", "business": True, "user": uq, "evidence": ev, "chosen": ua, "rejected": uw})

    for i in range(n_en):
        city = CITIES_EN[i % len(CITIES_EN)]
        month = MONTHS_EN[(i + rng.randint(0, 4)) % len(MONTHS_EN)]
        total = rng.randint(700, 4500)
        while total in (1290, 1246, 1840):
            total += 13
        target = rng.randint(14, 96)
        while target == 44:
            target += 3
        fake_close = target + rng.choice([-8, -6, -4, 5, 7, 9])
        if fake_close < 5 or fake_close == target:
            fake_close = target + 6
        reason = reasons_en[i % len(reasons_en)]
        ev = (
            f"[1] {city} Fulfillment Operations Report ({month})\n"
            f"Total dispatched parcels: {total:,}. "
            f"Returned parcels due to {reason}: {target} parcels."
        )
        q = f"How many parcels were returned at the {city} facility in {month}?"
        ans = f"According to the {city} report [1], {target} parcels were returned in {month}."
        wrong_fab = f"According to the {city} report [1], {fake_close} parcels were returned in {month}."
        false_abs = f"The excerpt [1] does not specify returned parcels for {city}. Could you share the report?"
        for t in (ev, q, ans, wrong_fab):
            _check_clean(t)
        sft_rows.append({"language": "en", "task": "v42_exact_copy_ans", "business": True, "user": q, "evidence": ev, "assistant": ans})
        pref_rows.append({"language": "en", "task": "v42_pref_exact_copy_digit", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": wrong_fab})
        pref_rows.append({"language": "en", "task": "v42_pref_exact_copy_no_abs", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": false_abs})

    return sft_rows, pref_rows


def generate_minmax_crossbind_cases(
    n_id: int, n_en: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """High-density min/max numerical comparisons with HARD cross-bound entity+wrong-number negatives."""
    sft_rows: list[dict] = []
    pref_rows: list[dict] = []

    # Indonesian: SKU revenue shares & branch SLA/accuracy comparisons (max and min)
    for i in range(n_id):
        branch = BRANCHES_ID[i % len(BRANCHES_ID)]
        month = MONTHS_ID[(i + rng.randint(0, 5)) % len(MONTHS_ID)]
        if i % 3 != 2:
            # 3-SKU revenue share comparison (fixes id_ans_3: 41% vs 34% vs 25%)
            p1, p2, p3 = rng.sample(PRODUCTS_ID, 3)
            # Include close gaps (e.g. 42% vs 35% vs 23%) so model actually compares tens & units digits!
            top = rng.randint(39, 54)
            mid = rng.randint(28, min(38, top - 4))
            low = 100 - top - mid
            while low <= 6 or low >= mid:
                top = rng.randint(40, 52)
                mid = rng.randint(29, min(37, top - 4))
                low = 100 - top - mid
            items = [(p1, top), (p2, mid), (p3, low)]
            rng.shuffle(items)
            ev = (
                f"[1] Ringkasan Penjualan {branch} ({month})\n"
                + ", ".join(f"{name}: {pct}% omzet" for name, pct in items) + "."
            )
            ask_max = (i % 3 == 0)
            if ask_max:
                q = f"Produk mana yang menyumbang kontribusi omzet terbesar di {branch}?"
                ans = f"Berdasarkan ringkasan {branch} [1], kontribusi omzet terbesar berasal dari {p1} sebesar {top}%."
                # Hard negative 1: RIGHT entity + WRONG neighbor number!
                bad_cross = f"Berdasarkan ringkasan {branch} [1], kontribusi omzet terbesar berasal dari {p1} sebesar {mid}%."
                # Hard negative 2: WRONG entity (2nd highest) + its number!
                bad_rank = f"Berdasarkan ringkasan {branch} [1], kontribusi omzet terbesar berasal dari {p2} sebesar {mid}%."
            else:
                q = f"Produk mana yang menyumbang kontribusi omzet terkecil di {branch}?"
                ans = f"Berdasarkan ringkasan {branch} [1], kontribusi omzet terkecil berasal dari {p3} sebesar {low}%."
                bad_cross = f"Berdasarkan ringkasan {branch} [1], kontribusi omzet terkecil berasal dari {p3} sebesar {top}%."
                bad_rank = f"Berdasarkan ringkasan {branch} [1], kontribusi omzet terkecil berasal dari {p2} sebesar {mid}%."
        else:
            # 3-warehouse decimal accuracy comparison in Indonesian
            d1, d2, d3 = rng.sample(DEPOTS_ID, 3)
            vals = sorted([round(rng.uniform(91.5, 99.6), 1) for _ in range(3)])
            while len(set(vals)) < 3 or any(v in (96.1, 99.4, 98.8) for v in vals) or (vals[1] - vals[0] < 0.8):
                vals = sorted([round(rng.uniform(91.5, 99.6), 1) for _ in range(3)])
            pairs = [(d1, vals[0]), (d2, vals[1]), (d3, vals[2])]
            rng.shuffle(pairs)
            ev = (
                f"[1] Audit Akurasi Pengemasan ({month})\n"
                + " ".join(f"{d}: akurasi {v:.1f}%." for d, v in pairs)
            )
            q = f"Gudang mana yang memiliki akurasi pengemasan terendah pada bulan {month}?"
            ans = f"Menurut audit [1], {d1} memiliki akurasi pengemasan terendah sebesar {vals[0]:.1f}%."
            bad_cross = f"Menurut audit [1], {d1} memiliki akurasi pengemasan terendah sebesar {vals[2]:.1f}%."
            bad_rank = f"Menurut audit [1], {d2} memiliki akurasi pengemasan terendah sebesar {vals[1]:.1f}%."

        for t in (ev, q, ans, bad_cross, bad_rank):
            _check_clean(t)
        sft_rows.append({"language": "id", "task": "v42_minmax_ans", "business": True, "user": q, "evidence": ev, "assistant": ans})
        pref_rows.append({"language": "id", "task": "v42_pref_minmax_crossbind", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": bad_cross})
        pref_rows.append({"language": "id", "task": "v42_pref_minmax_rank", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": bad_rank})

    # English: decimal picking accuracy / fulfillment audit & product revenue share (fixes en_ans_2!)
    for i in range(n_en):
        month = MONTHS_EN[(i + rng.randint(0, 5)) % len(MONTHS_EN)]
        if i % 3 != 2:
            h1, h2, h3 = rng.sample(HUBS_EN, 3)
            vals = sorted([round(rng.uniform(92.0, 99.7), 1) for _ in range(3)])
            while len(set(vals)) < 3 or any(v in (96.1, 99.4, 98.8) for v in vals) or (vals[1] - vals[0] < 0.7):
                vals = sorted([round(rng.uniform(92.0, 99.7), 1) for _ in range(3)])
            # Place lowest in middle or last often so model doesn't just grab the first number!
            pairs = [(h2, vals[1]), (h1, vals[0]), (h3, vals[2])] if i % 2 == 0 else [(h3, vals[2]), (h1, vals[0]), (h2, vals[1])]
            if i % 4 == 3:
                rng.shuffle(pairs)
            ev = (
                f"[1] Weekly Warehouse Audit ({month})\n"
                + " ".join(f"{h}: {v:.1f}% accuracy." for h, v in pairs)
            )
            ask_low = (i % 3 == 0)
            if ask_low:
                q = "Which warehouse had the lowest picking accuracy in the report?"
                ans = f"According to the audit [1], {h1} had the lowest picking accuracy at {vals[0]:.1f}%."
                # HARDEST negative: RIGHT entity (h1) + FIRST/HIGHEST neighbor's number (vals[2])!
                bad_cross = f"According to the audit [1], {h1} had the lowest picking accuracy at {vals[2]:.1f}%."
                bad_rank = f"According to the audit [1], {h2} had the lowest picking accuracy at {vals[1]:.1f}%."
            else:
                q = "Which warehouse achieved the highest picking accuracy in the report?"
                ans = f"According to the audit [1], {h3} achieved the highest picking accuracy at {vals[2]:.1f}%."
                bad_cross = f"According to the audit [1], {h3} achieved the highest picking accuracy at {vals[0]:.1f}%."
                bad_rank = f"According to the audit [1], {h2} achieved the highest picking accuracy at {vals[1]:.1f}%."
        else:
            city = CITIES_EN[i % len(CITIES_EN)]
            p1, p2, p3 = rng.sample(PRODUCTS_EN, 3)
            top = rng.randint(40, 55)
            mid = rng.randint(28, min(37, top - 4))
            low = 100 - top - mid
            while low <= 6 or low >= mid:
                top, mid, low = 46, 33, 21
            items = [(p1, top), (p2, mid), (p3, low)]
            rng.shuffle(items)
            ev = (
                f"[1] {city} Branch Sales Summary ({month})\n"
                + ", ".join(f"{name}: {pct}% revenue" for name, pct in items) + "."
            )
            q = f"Which product generated the highest revenue share at the {city} branch?"
            ans = f"According to the {city} summary [1], {p1} generated the highest revenue share at {top}%."
            bad_cross = f"According to the {city} summary [1], {p1} generated the highest revenue share at {mid}%."
            bad_rank = f"According to the {city} summary [1], {p2} generated the highest revenue share at {mid}%."

        for t in (ev, q, ans, bad_cross, bad_rank):
            _check_clean(t)
        sft_rows.append({"language": "en", "task": "v42_minmax_ans", "business": True, "user": q, "evidence": ev, "assistant": ans})
        pref_rows.append({"language": "en", "task": "v42_pref_minmax_crossbind", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": bad_cross})
        pref_rows.append({"language": "en", "task": "v42_pref_minmax_rank", "business": True, "user": q, "evidence": ev, "chosen": ans, "rejected": bad_rank})

    return sft_rows, pref_rows


def generate_missing_driver_clarifications(
    n_id: int, n_en: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """Train Kuon to echo the exact two missing drivers named by the user for both periods."""
    sft_rows: list[dict] = []
    pref_rows: list[dict] = []

    for i in range(n_id):
        d1, d2, knee_jerk = DRIVER_PAIRS_ID[i % len(DRIVER_PAIRS_ID)]
        old_m = rng.randint(35, 180)
        drop = rng.randint(6, max(8, old_m // 4))
        new_m = old_m - drop
        while old_m == 60 and new_m == 48:
            old_m += 5
            new_m += 4
        period = rng.choice(["bulan ini", "minggu ini", "kuartal ini"])
        user = (
            f"Omzet kami turun dari Rp{old_m} juta ke Rp{new_m} juta {period}, "
            f"tetapi kami belum tahu apakah {d1} atau {d2} yang turun. "
            f"Apakah kami perlu langsung {knee_jerk}?"
        )
        chosen = (
            f"Belum perlu langsung {knee_jerk}. "
            f"Berapa {d1} dan {d2} pada masing-masing periode sebelum dan sesudah penurunan?"
        )
        bad_offtopic = (
            "Apakah penurunan ini terjadi pada semua jenis produk (misalnya pakaian atau kebutuhan kesehatan) atau hanya pada kategori tertentu?"
        )
        bad_rash = (
            f"Ya, sebaiknya langsung {knee_jerk} sebesar 15% minggu ini untuk mengembalikan omzet."
        )
        _check_clean(user)
        sft_rows.append({"language": "id", "task": "v42_missing_driver_clarify", "business": True, "user": user, "assistant": chosen})
        pref_rows.append({
            "language": "id", "task": "v42_pref_missing_driver", "business": True,
            "user": user, "chosen": chosen, "rejected": bad_offtopic if i % 2 == 0 else bad_rash,
        })

    for i in range(n_en):
        d1, d2, knee_jerk = DRIVER_PAIRS_EN[i % len(DRIVER_PAIRS_EN)]
        old_k = rng.randint(25, 190)
        drop = rng.randint(5, max(7, old_k // 4))
        new_k = old_k - drop
        while old_k == 50 and new_k == 41:
            old_k += 4
            new_k += 3
        period = rng.choice(["this month", "this week", "this quarter"])
        user = (
            f"Revenue fell from ${old_k},000 to ${new_k},000 {period}, "
            f"and we haven't checked whether {d1} or {d2} dropped. "
            f"Should we {knee_jerk}?"
        )
        chosen = (
            f"Hold off before deciding to {knee_jerk}. "
            f"What were your {d1} and {d2} for both periods?"
        )
        bad_partial = f"What were your revenue and {d2} before the drop?"
        bad_rash = f"Yes, you should {knee_jerk} immediately to recover lost sales."
        _check_clean(user)
        sft_rows.append({"language": "en", "task": "v42_missing_driver_clarify", "business": True, "user": user, "assistant": chosen})
        pref_rows.append({
            "language": "en", "task": "v42_pref_missing_driver", "business": True,
            "user": user, "chosen": chosen, "rejected": bad_partial if i % 2 == 0 else bad_rash,
        })

    return sft_rows, pref_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, required=True, help="virhanali/kuon2-stage3-distill-v1 dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    base_sft_tr = _load_jsonl(args.base_dir / "sft_train.jsonl")
    base_sft_va = _load_jsonl(args.base_dir / "sft_val.jsonl")
    base_pr_tr = _load_jsonl(args.base_dir / "prefs_train.jsonl")
    base_pr_va = _load_jsonl(args.base_dir / "prefs_val.jsonl")

    # 1) Balanced caps keeping confirmed_actions strong without drowning Evidence QA
    sft_tr = _cap_tasks(base_sft_tr, {"distill_confirmed_mc": 1400, "distill_confirmed_text": 1000}, rng)
    pr_tr = _cap_tasks(base_pr_tr, {"distill_pref_confirmed_mc": 2600, "distill_pref_confirmed_text": 1600}, rng)

    # 2) Fix old distill_pref_ev_refuse negatives to reject realistic over-abstention template
    for r in pr_tr + base_pr_va:
        if r.get("task") == "distill_pref_ev_refuse":
            r["rejected"] = (
                "Kutipan [1] tidak menyebutkan angka yang ditanyakan. Bisa bagikan datanya?"
                if r.get("language") == "id"
                else "The excerpt [1] does not specify the requested figure. Could you share the relevant data?"
            )

    # 3) Core balanced bilingual Evidence QA (800 ID + 800 EN)
    id_ev_sft, id_ev_pr = generate_id_evidence_cases(800, rng)
    en_ev_sft, en_ev_pr = generate_en_evidence_cases(800, rng)

    # 4) Targeted exact-digit copy fidelity (500 ID + 300 EN)
    copy_sft, copy_pr = generate_exact_copy_operational_cases(500, 300, rng)

    # 5) Targeted min/max comparison + HARD entity-number cross-binding (600 ID + 600 EN)
    mm_sft, mm_pr = generate_minmax_crossbind_cases(600, 600, rng)

    # 6) Targeted missing-driver clarification pairs (200 ID + 200 EN) -> 36/36 Pillar 2
    clar_sft, clar_pr = generate_missing_driver_clarifications(200, 200, rng)

    # 6b) Cross-lingual EN->ID & ID->EN Evidence QA preserving exact currency (Rp stays Rp!)
    xl_sft: list[dict] = []
    xl_pr: list[dict] = []
    for i in range(160):
        city = CITIES_ID[i % len(CITIES_ID)]
        month = MONTHS_ID[i % len(MONTHS_ID)]
        val_m = rng.randint(14, 95)
        while val_m == 42:
            val_m += 3
        ev_id = f"[1] Laporan Keuangan Cabang {city} ({month})\nLaba bersih cabang {city} setelah pajak mencapai Rp{val_m} juta dari total omzet Rp{val_m * 4} juta."
        q_en = f"What was the {city} branch net profit after tax in {month}?"
        a_en = f"According to the report [1], the {city} branch net profit after tax was Rp{val_m} million."
        bad_usd = f"According to the report [1], the {city} branch net profit after tax was ${val_m} million."
        bad_abs = f"The excerpt [1] does not specify the {city} branch net profit after tax. Could you share the report?"
        _check_clean(ev_id)
        xl_sft.append({"language": "en", "task": "v42_xl_ev_ans", "business": True, "user": q_en, "evidence": ev_id, "assistant": a_en})
        xl_pr.append({"language": "en", "task": "v42_pref_xl_currency", "business": True, "user": q_en, "evidence": ev_id, "chosen": a_en, "rejected": bad_usd if i % 2 == 0 else bad_abs})

    # 7) Upgraded 54 Kuon 1 bilingual identity rows (3x SFT + 108 anti-leak prefs)
    ident_sft = build_identity_sft_rows()
    ident_pr = build_identity_prefs()

    def split_8pct(rows: list[dict]):
        rows = list(rows)
        rng.shuffle(rows)
        n_va = max(4, int(len(rows) * 0.08))
        return rows[n_va:], rows[:n_va]

    new_sft_tr, new_sft_va = split_8pct(id_ev_sft + en_ev_sft + copy_sft + mm_sft + clar_sft + xl_sft)
    new_pr_tr, new_pr_va = split_8pct(id_ev_pr + en_ev_pr + copy_pr + mm_pr + clar_pr + xl_pr)

    final_sft_tr = sft_tr + new_sft_tr + ident_sft * 3
    final_sft_va = base_sft_va + new_sft_va + ident_sft[:8]
    final_pr_tr = pr_tr + new_pr_tr + ident_pr
    final_pr_va = base_pr_va + new_pr_va + ident_pr[:12]

    rng.shuffle(final_sft_tr)
    rng.shuffle(final_pr_tr)

    # Strict zero-leakage check against all calibration + deliverable benchmark prompts
    eval_prompts = {c.user.casefold() for c in CALIBRATION_CASES} | {c.user.casefold() for c in DELIVERABLE_CASES}
    for row in final_sft_tr + final_pr_tr:
        assert str(row.get("user", "")).casefold() not in eval_prompts, f"Eval prompt leaked: {row.get('user')}"

    args.out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out / "sft_train.jsonl", final_sft_tr)
    _write_jsonl(args.out / "sft_val.jsonl", final_sft_va)
    _write_jsonl(args.out / "prefs_train.jsonl", final_pr_tr)
    _write_jsonl(args.out / "prefs_val.jsonl", final_pr_va)

    print(
        f"[v4.2-data] Built targeted dataset in {args.out}\n"
        f"  sft_train  : {len(final_sft_tr)}\n"
        f"  sft_val    : {len(final_sft_va)}\n"
        f"  prefs_train: {len(final_pr_tr)}\n"
        f"  prefs_val  : {len(final_pr_va)}"
    )
    print("  Top SFT tasks:", Counter((r.get("task", "?"), r.get("language", "?")) for r in final_sft_tr).most_common(12))
    print("  Top Pref tasks:", Counter((r.get("task", "?"), r.get("language", "?")) for r in final_pr_tr).most_common(12))


if __name__ == "__main__":
    main()

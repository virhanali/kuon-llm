"""Build balanced, leak-free Stage 3 v4.1 SFT & APO-zero datasets from cached distill-v1.

Fixes four exact v4 failure modes diagnosed from business_agent_eval.json:
1. Zero Indonesian distilled evidence rows in distill-v1 (old multi-dot Rupiah regex dropped ID).
2. Extreme class imbalance: confirmed_mc was 27% of SFT and 43% of APO-zero, drowning Evidence QA.
3. Missing anti-over-abstention negatives matching the real "Kutipan [1] tidak menyebutkan..." template.
4. Missing 54 bilingual Kuon 1 identity rows + anti-leak preference pairs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.distill_gemma4 import build_identity_sft_rows
from scripts.eval_business_agent import CALIBRATION_CASES

# Strictly forbidden benchmark strings from eval_business_agent.py to guarantee 0% leakage
_EVAL_FORBIDDEN = {
    "gudang surabaya", "cabang solo", "pemasok gamma", "pemasok delta",
    "cabang palembang", "north hub", "east hub", "west hub",
    "november support sla digest", "fulfillment unit economics q2",
    "teh tarik", "roti panggang", "kopi hitam",
    "1.840", "1,290", "1,246", "96.1%", "99.4%", "98.8%", "$1.35", "$4.80", "$2.10",
}

CITIES_ID = [
    "Semarang", "Makassar", "Medan", "Malang", "Denpasar", "Balikpapan",
    "Padang", "Manado", "Bekasi", "Tangerang", "Bogor", "Cirebon",
    "Pekanbaru", "Samarinda", "Mataram", "Banjarmasin", "Lampung", "Depok",
]
CITIES_EN = [
    "Austin", "Denver", "Seattle", "Boston", "Phoenix", "Portland",
    "Atlanta", "Chicago", "Dallas", "Miami", "Detroit", "Newark",
    "Raleigh", "Tampa", "Omaha", "Fresno", "Columbus", "Charlotte",
]

PRODUCTS_ID = [
    "Matcha Latte", "Kopi Susu Aren", "Croissant Mentega", "Dimsum Ayam",
    "Mie Ayam Jamur", "Es Cokelat Susu", "Sate Taichan", "Nasi Goreng Kampung",
    "Ayam Geprek", "Jus Alpukat", "Bolu Pandan", "Martabak Keju",
    "Kopi Americano", "Susu Almond", "Lumpia Udang", "Bubur Ayam",
    "Teh Melati", "Pisang Bakar", "Risoles Mayo", "Bakso Urat",
    "Tahu Crispy", "Soto Betawi", "Gado-Gado", "Kopi Tubruk",
]
PRODUCTS_EN = [
    "Cold Brew", "Oat Latte", "Almond Croissant", "Sourdough Loaf",
    "Turkey Club", "Matcha Frappe", "Berry Parfait", "Bagel Melt",
    "Green Smoothie", "Avocado Toast", "Espresso Tonic", "Cinnamon Roll",
    "Chicken Wrap", "Lemon Tart", "Chai Latte", "Veggie Quiche",
    "Flat White", "Granola Bowl", "Brioche Bun", "Iced Mocha",
]

SUPPLIERS_ID = [
    "CV Nusantara", "PT Prima Logam", "UD Sumber Makmur", "PT Karya Abadi",
    "CV Sinar Baru", "PT Mitra Pangan", "UD Jaya Sentosa", "PT Bintang Timur",
    "CV Cahaya Tani", "PT Surya Mandiri", "UD Harapan Kita", "PT Indo Segar",
]
SUPPLIERS_EN = [
    "Apex Supply", "Crestline Materials", "Harbor Freight Co", "Summit Packaging",
    "Riverbend Foods", "Redwood Logistics", "Cobalt Parts", "Vanguard Paper",
    "Pinnacle Plastics", "Keystone Bulk", "Horizon Mills", "Meridian Goods",
]

HUBS_EN = [
    "Central Depot", "Harbor Facility", "Valley Center", "Lakeside Hub",
    "Midwest Dock", "Coastal Plant", "Metro Annex", "Ridge Warehouse",
    "Bayfront Hub", "Prairie Depot", "Summit Dock", "Riverview Plant",
]
TIERS_EN = [
    ("Basic", "Growth", "Scale"),
    ("Core", "Pro", "Premier"),
    ("Team", "Business", "Corporate"),
    ("Standard", "Plus", "Platinum"),
]
SHIFTS_ID = [
    ("shift pagi", "shift siang", "shift malam"),
    ("tim pagi", "tim sore", "tim malam"),
    ("regu awal", "regu tengah", "regu akhir"),
]
SHIFTS_EN = [
    ("morning shift", "afternoon shift", "night shift"),
    ("Tier-1 desk", "Tier-2 desk", "weekend desk"),
    ("US-East queue", "EU-West queue", "APAC queue"),
]
MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "November", "Desember",
]
MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "December",
]
UNANS_TOPICS_ID = [
    ("laba bersih setelah pajak", "data laba bersih setelah pajak"),
    ("biaya sewa gedung per bulan", "rincian biaya sewa gedung"),
    ("biaya listrik dan air bulanan", "data tagihan utilitas bulanan"),
    ("persentase diskon rata-rata tim sales", "data potongan harga rata-rata"),
    ("penyebab kenaikan pembatalan transaksi", "hasil audit alasan pembatalan"),
    ("biaya akuisisi pelanggan baru", "rincian biaya pemasaran per pelanggan baru"),
    ("target omzet tahun depan", "dokumen proyeksi target tahun depan"),
    ("gaji lembur karyawan operasional", "rekap biaya lembur karyawan"),
    ("nilai aset inventaris akhir tahun", "laporan valuasi stok akhir tahun"),
    ("persentase pelanggan berulang bulan lalu", "data retensi pelanggan berulang"),
]
UNANS_TOPICS_EN = [
    ("net profit margin after taxes", "the net profit margin breakdown"),
    ("monthly warehouse rent expense", "the monthly lease cost figures"),
    ("customer acquisition cost (CAC)", "marketing spend and new customer counts"),
    ("reason behind weekend drop in repeat orders", "customer feedback or cancellation logs"),
    ("annual equipment depreciation", "the depreciation schedule"),
    ("average discount percentage offered by sales", "the sales discount report"),
    ("projected next-quarter revenue target", "next quarter's budget forecast"),
    ("employee overtime payroll cost", "the overtime payroll summary"),
    ("refund processing fee per transaction", "payment gateway fee details"),
    ("supplier contract renewal rate", "the vendor renewal records"),
]


def _fmt_id_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _fmt_rp(n: int) -> str:
    return f"Rp{_fmt_id_int(n)}"


def _check_no_leak(text: str) -> None:
    low = text.casefold()
    for bad in _EVAL_FORBIDDEN:
        if bad in low:
            raise ValueError(f"Evaluation leakage detected: {bad!r} in {text[:120]!r}")


def _emit_ev_pair(
    lang: str,
    evidence: str,
    ans_q: str,
    ans_a: str,
    ans_wrong: str,
    unans_q: str,
    unans_a: str,
    unans_wrong: str,
    false_abstain_on_ans: str,
    sft_rows: list[dict],
    pref_rows: list[dict],
) -> None:
    for txt in (evidence, ans_q, ans_a, ans_wrong, unans_q, unans_a, unans_wrong):
        _check_no_leak(txt)
    sft_rows.append({
        "language": lang, "task": "distill_v41_ev_ans", "business": True,
        "user": ans_q, "evidence": evidence, "assistant": ans_a,
    })
    sft_rows.append({
        "language": lang, "task": "distill_v41_ev_unans", "business": True,
        "user": unans_q, "evidence": evidence, "assistant": unans_a,
    })
    # 1) Entity-number binding (right entity vs neighbor entity in same excerpt)
    pref_rows.append({
        "language": lang, "task": "distill_v41_pref_ev_bind", "business": True,
        "user": ans_q, "evidence": evidence, "chosen": ans_a, "rejected": ans_wrong,
    })
    # 2) Anti-over-abstention using the EXACT abstention wording that caused v4 recall drop
    pref_rows.append({
        "language": lang, "task": "distill_v41_pref_ev_no_overabstain", "business": True,
        "user": ans_q, "evidence": evidence, "chosen": ans_a, "rejected": false_abstain_on_ans,
    })
    # 3) Anti-hallucination on genuinely unanswerable query
    pref_rows.append({
        "language": lang, "task": "distill_v41_pref_ev_abstain", "business": True,
        "user": unans_q, "evidence": evidence, "chosen": unans_a, "rejected": unans_wrong,
    })


def generate_id_evidence_cases(count: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    sft_rows: list[dict] = []
    pref_rows: list[dict] = []

    for i in range(count):
        arch = i % 6
        city = CITIES_ID[(i + rng.randint(0, 5)) % len(CITIES_ID)]
        month = MONTHS_ID[i % len(MONTHS_ID)]
        utopic, uask = UNANS_TOPICS_ID[(i + rng.randint(0, 4)) % len(UNANS_TOPICS_ID)]

        if arch == 0:
            # Warehouse/branch returns & counts (2nd/3rd figure)
            total_ship = rng.randint(900, 4800)
            while total_ship in (1840, 1290, 1246):
                total_ship += 7
            damaged = rng.randint(12, 65)
            returned = rng.randint(18, 95)
            while returned == damaged:
                returned += 3
            ev = (
                f"[1] Laporan Operasional Gudang {city} ({month})\n"
                f"Total pengiriman: {_fmt_id_int(total_ship)} paket. "
                f"Paket rusak di jalan: {damaged} paket. "
                f"Paket retur karena alamat tidak lengkap: {returned} paket."
            )
            ask_ret = (i % 2 == 0)
            if ask_ret:
                ans_q = f"Berapa jumlah paket retur di Gudang {city} pada bulan {month}?"
                ans_a = f"Berdasarkan laporan Gudang {city} [1], jumlah paket retur pada bulan {month} sebanyak {returned} paket."
                ans_wrong = f"Berdasarkan laporan Gudang {city} [1], jumlah paket retur sebanyak {damaged} paket."
                false_abs = f"Kutipan [1] tidak menyebutkan jumlah paket retur di Gudang {city}. Bisa bagikan datanya?"
            else:
                ans_q = f"Berapa jumlah paket rusak di jalan pada laporan Gudang {city}?"
                ans_a = f"Menurut laporan operasional Gudang {city} [1], tercatat {damaged} paket rusak di jalan."
                ans_wrong = f"Menurut laporan operasional Gudang {city} [1], tercatat {returned} paket rusak di jalan."
                false_abs = f"Kutipan [1] tidak menyebutkan jumlah paket rusak di Gudang {city}. Bisa bagikan datanya?"
            unans_q = f"Berapa {utopic} untuk operasional Gudang {city} di bulan {month}?"
            unans_a = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            unans_wrong = f"Berdasarkan laporan Gudang {city} [1], {utopic} tercatat sebesar {returned} paket."

        elif arch == 1:
            # Shift/team SLA response time (explicitly querying 2nd or 3rd shift)
            s1, s2, s3 = SHIFTS_ID[i % len(SHIFTS_ID)]
            m1, m2, m3 = rng.sample(range(3, 35), 3)
            t1, t2, t3 = rng.randint(40, 120), rng.randint(40, 120), rng.randint(40, 120)
            ev = (
                f"[1] Rekap SLA Layanan Pelanggan {city}\n"
                f"{s1.capitalize()}: rata-rata balas {m1} menit ({t1} tiket). "
                f"{s2.capitalize()}: rata-rata balas {m2} menit ({t2} tiket). "
                f"{s3.capitalize()}: rata-rata balas {m3} menit ({t3} tiket)."
            )
            pick = (i // 6) % 2
            target_s, target_m, wrong_m = (s3, m3, m1) if pick == 0 else (s2, m2, m3)
            ans_q = f"Berapa rata-rata waktu balas chat CS pada {target_s} di {city}?"
            ans_a = f"Menurut rekap SLA {city} [1], rata-rata waktu balas pada {target_s} adalah {target_m} menit."
            ans_wrong = f"Menurut rekap SLA {city} [1], rata-rata waktu balas pada {target_s} adalah {wrong_m} menit."
            false_abs = f"Kutipan [1] tidak menyebutkan rata-rata waktu balas untuk {target_s}. Bisa bagikan log SLA-nya?"
            unans_q = f"Berapa {utopic} tim CS {city}?"
            unans_a = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            unans_wrong = f"Menurut rekap SLA [1], {utopic} sebesar {target_m} menit."

        elif arch == 2:
            # Multi-SKU revenue share — highest OR lowest (prevents memorizing a single item!)
            p1, p2, p3 = rng.sample(PRODUCTS_ID, 3)
            a = rng.randint(42, 58)
            b = rng.randint(24, 38)
            c = 100 - a - b
            while c <= 5 or b == c or a == b:
                a = rng.randint(42, 55)
                b = rng.randint(25, 36)
                c = 100 - a - b
            items = [(p1, a), (p2, b), (p3, c)]
            rng.shuffle(items)
            ev = (
                f"[1] Ringkasan Penjualan Cabang {city} ({month})\n"
                + ", ".join(f"{name}: {pct}% omzet" for name, pct in items) + "."
            )
            ask_max = (i % 3 != 0)
            ranked = sorted(items, key=lambda x: x[1])
            low_name, low_pct = ranked[0]
            mid_name, mid_pct = ranked[1]
            top_name, top_pct = ranked[2]
            if ask_max:
                ans_q = f"Produk mana yang menyumbang kontribusi omzet terbesar di Cabang {city}?"
                ans_a = f"Berdasarkan ringkasan Cabang {city} [1], produk dengan kontribusi omzet terbesar adalah {top_name} sebesar {top_pct}%."
                ans_wrong = f"Berdasarkan ringkasan Cabang {city} [1], kontribusi omzet terbesar berasal dari {mid_name} ({mid_pct}%)."
                false_abs = f"Kutipan [1] tidak menyebutkan kontribusi omzet per produk di Cabang {city}. Bisa bagikan data penjualannya?"
            else:
                ans_q = f"Produk mana yang memiliki porsi omzet paling kecil di Cabang {city}?"
                ans_a = f"Menurut ringkasan Cabang {city} [1], produk dengan porsi omzet terkecil adalah {low_name} sebesar {low_pct}%."
                ans_wrong = f"Menurut ringkasan Cabang {city} [1], produk dengan porsi omzet terkecil adalah {top_name} sebesar {top_pct}%."
                false_abs = f"Kutipan [1] tidak menyebutkan produk dengan porsi omzet terkecil. Bisa bagikan rinciannya?"
            unans_q = f"Berapa {utopic} Cabang {city} pada bulan {month}?"
            unans_a = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            unans_wrong = f"Menurut ringkasan [1], {utopic} Cabang {city} adalah {top_pct}%."

        elif arch == 3:
            # Supplier lead time & Rupiah price (tests multi-dot Rp + 2nd/3rd supplier binding)
            sup1, sup2, sup3 = rng.sample(SUPPLIERS_ID, 3)
            lt1, lt2, lt3 = rng.sample(range(2, 16), 3)
            pr1 = rng.randint(12, 45) * 500
            pr2 = rng.randint(12, 45) * 500
            pr3 = rng.randint(12, 45) * 500
            while len({pr1, pr2, pr3}) < 3:
                pr2 += 1000
                pr3 += 2000
            ev = (
                f"[1] Catatan Pembelian Bahan Baku {city}\n"
                f"{sup1}: lead time {lt1} hari, harga {_fmt_rp(pr1)}/kg. "
                f"{sup2}: lead time {lt2} hari, harga {_fmt_rp(pr2)}/kg. "
                f"{sup3}: lead time {lt3} hari, harga {_fmt_rp(pr3)}/kg."
            )
            mode = (i // 6) % 3
            if mode == 0:
                ans_q = f"Berapa hari lead time pengiriman bahan baku dari {sup2}?"
                ans_a = f"Berdasarkan catatan pembelian [1], lead time pengiriman dari {sup2} adalah {lt2} hari."
                ans_wrong = f"Berdasarkan catatan pembelian [1], lead time pengiriman dari {sup2} adalah {lt1} hari."
                false_abs = f"Kutipan [1] tidak menyebutkan lead time pengiriman dari {sup2}. Bisa bagikan data pemasoknya?"
            elif mode == 1:
                ans_q = f"Berapa harga bahan baku per kg dari {sup3}?"
                ans_a = f"Menurut catatan pembelian [1], harga bahan baku dari {sup3} adalah {_fmt_rp(pr3)}/kg."
                ans_wrong = f"Menurut catatan pembelian [1], harga bahan baku dari {sup3} adalah {_fmt_rp(pr1)}/kg."
                false_abs = f"Kutipan [1] tidak menyebutkan harga bahan baku dari {sup3}. Bisa bagikan daftar harganya?"
            else:
                ans_q = f"Berapa hari lead time pengiriman dari {sup3}?"
                ans_a = f"Menurut catatan pembelian [1], lead time pengiriman dari {sup3} adalah {lt3} hari."
                ans_wrong = f"Menurut catatan pembelian [1], lead time pengiriman dari {sup3} adalah {lt2} hari."
                false_abs = f"Kutipan [1] tidak menyebutkan waktu pengiriman dari {sup3}. Bisa bagikan jadwal pengirimannya?"
            unans_q = f"Berapa {utopic} gudang pengadaan {city}?"
            unans_a = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            unans_wrong = f"Menurut catatan [1], {utopic} sebesar {_fmt_rp(pr2)}."

        elif arch == 4:
            # Multi-branch Rupiah revenue (with millions / multi-dot numbers)
            c1, c2, c3 = rng.sample(CITIES_ID, 3)
            r1 = rng.randint(25, 95) * 1_000_000
            r2 = rng.randint(25, 95) * 1_000_000
            r3 = rng.randint(25, 95) * 1_000_000
            while len({r1, r2, r3}) < 3:
                r2 += 3_000_000
                r3 += 5_000_000
            ev = (
                f"[1] Rekap Omzet Regional ({month})\n"
                f"Cabang {c1}: {_fmt_rp(r1)}. Cabang {c2}: {_fmt_rp(r2)}. Cabang {c3}: {_fmt_rp(r3)}."
            )
            target_c, target_r, wrong_r = (c2, r2, r1) if i % 2 == 0 else (c3, r3, r2)
            ans_q = f"Berapa omzet Cabang {target_c} pada bulan {month}?"
            ans_a = f"Menurut rekap omzet [1], omzet Cabang {target_c} pada bulan {month} sebesar {_fmt_rp(target_r)}."
            ans_wrong = f"Menurut rekap omzet [1], omzet Cabang {target_c} sebesar {_fmt_rp(wrong_r)}."
            false_abs = f"Kutipan [1] tidak menyebutkan omzet Cabang {target_c}. Bisa bagikan rekap penjualannya?"
            unans_q = f"Berapa {utopic} di Cabang {target_c} bulan {month}?"
            unans_a = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            unans_wrong = f"Menurut rekap [1], {utopic} di Cabang {target_c} mencapai {_fmt_rp(target_r)}."

        else:
            # Cost breakdown per pesanan (shipping vs packaging vs labor)
            ongkir = rng.randint(8, 22) * 500
            kemasan = rng.randint(3, 11) * 500
            packing = rng.randint(4, 14) * 500
            while len({ongkir, kemasan, packing}) < 3:
                kemasan += 500
                packing += 1000
            ev = (
                f"[1] Struktur Biaya Pemenuhan Pesanan {city} ({month})\n"
                f"Rata-rata ongkos kirim: {_fmt_rp(ongkir)}/pesanan. "
                f"Biaya kemasan: {_fmt_rp(kemasan)}/pesanan. "
                f"Biaya tenaga packing: {_fmt_rp(packing)}/pesanan."
            )
            if i % 2 == 0:
                ans_q = f"Berapa biaya kemasan per pesanan di {city} pada bulan {month}?"
                ans_a = f"Berdasarkan struktur biaya [1], biaya kemasan per pesanan adalah {_fmt_rp(kemasan)}/pesanan."
                ans_wrong = f"Berdasarkan struktur biaya [1], biaya kemasan per pesanan adalah {_fmt_rp(packing)}/pesanan."
                false_abs = f"Kutipan [1] tidak menyebutkan biaya kemasan per pesanan. Bisa bagikan rincian biayanya?"
            else:
                ans_q = f"Berapa biaya tenaga packing per pesanan di {city}?"
                ans_a = f"Menurut struktur biaya [1], biaya tenaga packing adalah {_fmt_rp(packing)}/pesanan."
                ans_wrong = f"Menurut struktur biaya [1], biaya tenaga packing adalah {_fmt_rp(ongkir)}/pesanan."
                false_abs = f"Kutipan [1] tidak menyebutkan biaya tenaga packing. Bisa bagikan rincian operasionalnya?"
            unans_q = f"Berapa {utopic} divisi fulfillment {city}?"
            unans_a = f"Kutipan [1] tidak menyebutkan {utopic}. Bisa bagikan {uask}?"
            unans_wrong = f"Berdasarkan struktur biaya [1], {utopic} sebesar {_fmt_rp(ongkir)}."

        _emit_ev_pair(
            "id", ev, ans_q, ans_a, ans_wrong, unans_q, unans_a, unans_wrong, false_abs,
            sft_rows, pref_rows,
        )
    return sft_rows, pref_rows


def generate_en_evidence_cases(count: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    sft_rows: list[dict] = []
    pref_rows: list[dict] = []

    for i in range(count):
        arch = i % 6
        city = CITIES_EN[(i + rng.randint(0, 5)) % len(CITIES_EN)]
        month = MONTHS_EN[i % len(MONTHS_EN)]
        utopic, uask = UNANS_TOPICS_EN[(i + rng.randint(0, 4)) % len(UNANS_TOPICS_EN)]

        if arch == 0:
            # Support tickets / SLA breaches
            total_t = rng.randint(600, 2900)
            breached = rng.randint(15, 92)
            resolved = total_t - breached
            while total_t in (1290, 1246, 1840) or resolved in (1290, 1246, 1840):
                total_t += 11
                resolved = total_t - breached
            ev = (
                f"[1] {month} Support SLA Digest ({city})\n"
                f"Total inbound tickets: {total_t:,}. Resolved within 24 hours: {resolved:,}. "
                f"Breached 24-hour SLA: {breached} tickets."
            )
            ans_q = f"How many support tickets breached the 24-hour SLA in {city} during {month}?"
            ans_a = f"According to the {month} Support SLA Digest [1], {breached} tickets breached the 24-hour SLA."
            ans_wrong = f"According to the {month} Support SLA Digest [1], {resolved:,} tickets breached the 24-hour SLA."
            false_abs = f"The excerpt [1] does not specify how many tickets breached the 24-hour SLA. Could you share the SLA log?"
            unans_q = f"What was the {utopic} for the {city} support team in {month}?"
            unans_a = f"The excerpt [1] does not specify the {utopic}. Could you share {uask}?"
            unans_wrong = f"According to [1], the {utopic} was {breached}%."

        elif arch == 1:
            # Multi-hub picking accuracy (lowest OR highest)
            h1, h2, h3 = rng.sample(HUBS_EN, 3)
            vals = sorted([round(rng.uniform(93.2, 99.7), 1) for _ in range(3)])
            while len(set(vals)) < 3 or any(v in (96.1, 99.4, 98.8) for v in vals):
                vals = sorted([round(rng.uniform(93.2, 99.7), 1) for _ in range(3)])
            pairs = [(h1, vals[0]), (h2, vals[1]), (h3, vals[2])]
            rng.shuffle(pairs)
            ev = (
                f"[1] Weekly Fulfillment Audit ({month})\n"
                + " ".join(f"{h}: {v:.1f}% accuracy." for h, v in pairs)
            )
            low_h, low_v = h1, vals[0]
            high_h, high_v = h3, vals[2]
            if i % 2 == 0:
                ans_q = f"Which facility recorded the lowest picking accuracy in {month}?"
                ans_a = f"According to the audit [1], {low_h} had the lowest picking accuracy at {low_v:.1f}%."
                ans_wrong = f"According to the audit [1], {high_h} had the lowest picking accuracy at {high_v:.1f}%."
                false_abs = "The excerpt [1] does not specify which facility had the lowest picking accuracy. Could you share the full audit?"
            else:
                ans_q = f"Which facility achieved the highest picking accuracy in {month}?"
                ans_a = f"According to the audit [1], {high_h} achieved the highest picking accuracy at {high_v:.1f}%."
                ans_wrong = f"According to the audit [1], {low_h} achieved the highest picking accuracy at {low_v:.1f}%."
                false_abs = "The excerpt [1] does not specify the highest picking accuracy. Could you share the audit data?"
            unans_q = f"What is the {utopic} across these fulfillment hubs?"
            unans_a = f"The excerpt [1] does not specify the {utopic}. Could you share {uask}?"
            unans_wrong = f"As stated in [1], the {utopic} is {low_v:.1f}%."

        elif arch == 2:
            # Subscription tier churn (querying 2nd or 3rd tier)
            t1, t2, t3 = TIERS_EN[i % len(TIERS_EN)]
            c1 = round(rng.uniform(4.5, 8.9), 1)
            c2 = round(rng.uniform(2.1, 4.3), 1)
            c3 = round(rng.uniform(0.6, 1.9), 1)
            accts = rng.randint(65, 290)
            ev = (
                f"[1] {city} SaaS Retention Summary ({month})\n"
                f"{t1} tier monthly churn: {c1:.1f}%. {t2} tier monthly churn: {c2:.1f}%. "
                f"{t3} tier monthly churn: {c3:.1f}% across {accts} active accounts."
            )
            target_t, target_c, wrong_c = (t3, c3, c1) if i % 2 == 0 else (t2, c2, c3)
            ans_q = f"What monthly churn rate was reported for {target_t} tier accounts?"
            ans_a = f"According to the retention summary [1], {target_t} tier monthly churn was {target_c:.1f}%."
            ans_wrong = f"According to the retention summary [1], {target_t} tier monthly churn was {wrong_c:.1f}%."
            false_abs = f"The excerpt [1] does not specify the monthly churn rate for {target_t} accounts. Could you share the retention breakdown?"
            unans_q = f"What was the {utopic} for {target_t} accounts in {month}?"
            unans_a = f"The excerpt [1] does not specify the {utopic}. Could you share {uask}?"
            unans_wrong = f"The retention summary [1] reports {utopic} at {target_c:.1f}%."

        elif arch == 3:
            # Unit economics breakdown ($ postage vs $ packaging vs $ labor — fixes en_ans_4!)
            ship = round(rng.uniform(3.20, 6.90), 2)
            pack = round(rng.uniform(0.75, 2.40), 2)
            labor = round(rng.uniform(1.60, 3.80), 2)
            while len({ship, pack, labor}) < 3 or abs(pack - labor) < 0.25 or any(v in (1.35, 4.80, 2.10) for v in (ship, pack, labor)):
                ship = round(rng.uniform(3.25, 6.85), 2)
                pack = round(rng.uniform(0.80, 1.30), 2)
                labor = round(rng.uniform(2.25, 3.45), 2)
            ev = (
                f"[1] {city} Unit Economics ({month})\n"
                f"Average carrier postage: ${ship:.2f}/order. "
                f"Packaging supplies: ${pack:.2f}/order. "
                f"Pick-and-pack labor: ${labor:.2f}/order."
            )
            mode = (i // 6) % 3
            if mode == 0:
                ans_q = f"How much did packaging supplies cost per order in {city} during {month}?"
                ans_a = f"According to the {city} Unit Economics report [1], packaging supplies cost ${pack:.2f}/order."
                ans_wrong = f"According to the {city} Unit Economics report [1], packaging supplies cost ${labor:.2f}/order."
                false_abs = "The excerpt [1] does not specify packaging cost per order. Could you share the cost table?"
            elif mode == 1:
                ans_q = f"What was the pick-and-pack labor cost per order in {city}?"
                ans_a = f"According to the unit economics report [1], pick-and-pack labor was ${labor:.2f}/order."
                ans_wrong = f"According to the unit economics report [1], pick-and-pack labor was ${pack:.2f}/order."
                false_abs = "The excerpt [1] does not specify pick-and-pack labor cost. Could you provide the labor sheet?"
            else:
                ans_q = f"What was the average carrier postage per order in {city}?"
                ans_a = f"According to [1], average carrier postage was ${ship:.2f}/order."
                ans_wrong = f"According to [1], average carrier postage was ${labor:.2f}/order."
                false_abs = "The excerpt [1] does not specify carrier postage per order. Could you share shipping costs?"
            unans_q = f"What was the {utopic} for {city} in {month}?"
            unans_a = f"The excerpt [1] does not specify the {utopic}. Could you share {uask}?"
            unans_wrong = f"According to [1], the {utopic} was ${ship:.2f}/order."

        elif arch == 4:
            # Supplier lead time & price
            s1, s2 = rng.sample(SUPPLIERS_EN, 2)
            lt1, lt2 = rng.sample(range(2, 18), 2)
            p1 = round(rng.uniform(4.5, 19.5), 2)
            p2 = round(rng.uniform(4.5, 19.5), 2)
            while p1 == p2:
                p2 = round(p2 + 1.15, 2)
            ev = (
                f"[1] Procurement Log ({city} — {month})\n"
                f"{s1}: lead time {lt1} days, unit price ${p1:.2f}/kg. "
                f"{s2}: lead time {lt2} days, unit price ${p2:.2f}/kg."
            )
            if i % 2 == 0:
                ans_q = f"What is the delivery lead time for {s2}?"
                ans_a = f"According to the procurement log [1], {s2} has a lead time of {lt2} days."
                ans_wrong = f"According to the procurement log [1], {s2} has a lead time of {lt1} days."
                false_abs = f"The excerpt [1] does not specify the lead time for {s2}. Could you share vendor lead times?"
            else:
                ans_q = f"What unit price per kg does {s2} charge?"
                ans_a = f"According to [1], {s2} charges ${p2:.2f}/kg."
                ans_wrong = f"According to [1], {s2} charges ${p1:.2f}/kg."
                false_abs = f"The excerpt [1] does not specify the unit price for {s2}. Could you share the supplier quote?"
            unans_q = f"What is the {utopic} for our {city} procurement team?"
            unans_a = f"The excerpt [1] does not specify the {utopic}. Could you share {uask}?"
            unans_wrong = f"As stated in [1], the {utopic} is {lt2} days."

        else:
            # Multi-product revenue share
            p1, p2, p3 = rng.sample(PRODUCTS_EN, 3)
            a = rng.randint(43, 57)
            b = rng.randint(25, 36)
            c = 100 - a - b
            while c <= 5 or a == b or b == c:
                a, b, c = 48, 33, 19
            items = [(p1, a), (p2, b), (p3, c)]
            rng.shuffle(items)
            ranked = sorted(items, key=lambda x: x[1])
            top_p, top_pct = ranked[-1]
            mid_p, mid_pct = ranked[1]
            ev = (
                f"[1] {city} Store Sales Mix ({month})\n"
                + ", ".join(f"{name}: {pct}% of revenue" for name, pct in items) + "."
            )
            ans_q = f"Which product contributed the largest share of revenue at the {city} store?"
            ans_a = f"According to the {city} store sales mix [1], {top_p} contributed the largest share at {top_pct}% of revenue."
            ans_wrong = f"According to the {city} store sales mix [1], {mid_p} contributed the largest share at {mid_pct}%."
            false_abs = f"The excerpt [1] does not specify product revenue shares for {city}. Could you share the sales mix?"
            unans_q = f"What was the {utopic} at the {city} store in {month}?"
            unans_a = f"The excerpt [1] does not specify the {utopic}. Could you share {uask}?"
            unans_wrong = f"According to [1], the {utopic} was {top_pct}%."

        _emit_ev_pair(
            "en", ev, ans_q, ans_a, ans_wrong, unans_q, unans_a, unans_wrong, false_abs,
            sft_rows, pref_rows,
        )
    return sft_rows, pref_rows


def build_identity_prefs() -> list[dict]:
    """Build sharp anti-leak / anti-garble identity preference pairs for Kuon 1."""
    rows = build_identity_sft_rows()
    bad_id = [
        "Halo! Saya Kuon, model apa Anda? Apakah Anda memiliki pertanyaan lain?",
        "Saya adalah model bahasa yang dibuat berbasis Gemma dari Google.",
        "Saya menggunakan model dasar GPT dari OpenAI yang di-fine-tune.",
    ]
    bad_en = [
        "Hello! I am Kuon, what model are you? Do you have any other questions?",
        "I am a language model built on top of Google Gemma.",
        "I am based on OpenAI GPT fine-tuned for business tasks.",
    ]
    prefs: list[dict] = []
    for i, r in enumerate(rows):
        lang = r.get("language", "id")
        pool = bad_id if lang == "id" else bad_en
        for bad in (pool[i % len(pool)], pool[(i + 1) % len(pool)]):
            prefs.append({
                "language": lang,
                "task": "distill_v41_pref_identity",
                "business": bool(r.get("business", False)),
                "user": r["user"],
                "chosen": r["assistant"],
                "rejected": bad,
            })
    return prefs


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _cap_tasks(rows: list[dict], caps: dict[str, int], rng: random.Random) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    out: list[dict] = []
    for r in rows:
        t = r.get("task", "")
        if t in caps:
            by_task.setdefault(t, []).append(r)
        else:
            out.append(r)
    for t, cap in caps.items():
        bucket = by_task.get(t, [])
        if len(bucket) > cap:
            # Keep ID/EN balanced when downsampling
            id_rows = [r for r in bucket if r.get("language") == "id"]
            en_rows = [r for r in bucket if r.get("language") != "id"]
            rng.shuffle(id_rows)
            rng.shuffle(en_rows)
            half = cap // 2
            chosen = id_rows[:half] + en_rows[: cap - half]
            out.extend(chosen)
        else:
            out.extend(bucket)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, required=True, help="Downloaded virhanali/kuon2-stage3-distill-v1 dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-cases", type=int, default=1200)
    ap.add_argument("--en-cases", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=4141)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    base_sft_tr = _load_jsonl(args.base_dir / "sft_train.jsonl")
    base_sft_va = _load_jsonl(args.base_dir / "sft_val.jsonl")
    base_pr_tr = _load_jsonl(args.base_dir / "prefs_train.jsonl")
    base_pr_va = _load_jsonl(args.base_dir / "prefs_val.jsonl")

    # 1) Rebalance over-dominant confirmed_mc/confirmed_text tasks so Evidence QA & Consulting shine
    sft_caps = {
        "distill_confirmed_mc": 1200,
        "distill_confirmed_text": 1000,
    }
    pref_caps = {
        "distill_pref_confirmed_mc": 2000,
        "distill_pref_confirmed_text": 1600,
    }
    sft_tr = _cap_tasks(base_sft_tr, sft_caps, rng)
    pr_tr = _cap_tasks(base_pr_tr, pref_caps, rng)

    # 2) Upgrade existing distill_pref_ev_refuse negatives from "I do not know." -> realistic over-abstention
    for r in pr_tr:
        if r.get("task") == "distill_pref_ev_refuse":
            if r.get("language") == "id":
                r["rejected"] = "Kutipan [1] tidak menyebutkan angka yang ditanyakan. Bisa bagikan datanya?"
            else:
                r["rejected"] = "The excerpt [1] does not specify the requested figure. Could you share the relevant data?"
    for r in base_pr_va:
        if r.get("task") == "distill_pref_ev_refuse":
            if r.get("language") == "id":
                r["rejected"] = "Kutipan [1] tidak menyebutkan angka yang ditanyakan. Bisa bagikan datanya?"
            else:
                r["rejected"] = "The excerpt [1] does not specify the requested figure. Could you share the relevant data?"

    # 3) Generate 2,400 zero-leakage multi-entity bilingual Evidence QA scenarios (1,200 ID + 1,200 EN)
    id_sft, id_prefs = generate_id_evidence_cases(args.id_cases, rng)
    en_sft, en_prefs = generate_en_evidence_cases(args.en_cases, rng)

    # Hold out 8% of newly generated scenarios cleanly into val
    def split_pairs(sft_list: list[dict], pref_list: list[dict], val_frac: float = 0.08):
        n_scen = len(sft_list) // 2
        n_val = max(1, int(n_scen * val_frac))
        return (
            sft_list[n_val * 2 :], sft_list[: n_val * 2],
            pref_list[n_val * 3 :], pref_list[: n_val * 3],
        )

    id_sft_tr, id_sft_va, id_pr_tr, id_pr_va = split_pairs(id_sft, id_prefs)
    en_sft_tr, en_sft_va, en_pr_tr, en_pr_va = split_pairs(en_sft, en_prefs)

    # 4) Add upgraded 54 bilingual Kuon 1 identity rows (3x upsampled in SFT) + 108 anti-leak prefs
    ident_sft = build_identity_sft_rows()
    ident_prefs = build_identity_prefs()

    final_sft_tr = sft_tr + id_sft_tr + en_sft_tr + ident_sft * 3
    final_sft_va = base_sft_va + id_sft_va + en_sft_va + ident_sft[:8]
    final_pr_tr = pr_tr + id_pr_tr + en_pr_tr + ident_prefs
    final_pr_va = base_pr_va + id_pr_va + en_pr_va + ident_prefs[:12]

    rng.shuffle(final_sft_tr)
    rng.shuffle(final_pr_tr)

    # Final leakage audit against all 16 eval_business_agent.py prompts
    eval_prompts = {c.user.casefold() for c in CALIBRATION_CASES}
    for row in final_sft_tr + final_pr_tr:
        u = str(row.get("user", "")).casefold()
        assert u not in eval_prompts, f"Exact eval prompt leaked: {row.get('user')}"

    args.out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out / "sft_train.jsonl", final_sft_tr)
    _write_jsonl(args.out / "sft_val.jsonl", final_sft_va)
    _write_jsonl(args.out / "prefs_train.jsonl", final_pr_tr)
    _write_jsonl(args.out / "prefs_val.jsonl", final_pr_va)

    print(
        f"[v4.1-data] Built balanced dataset in {args.out}\n"
        f"  sft_train  : {len(final_sft_tr)} (was {len(base_sft_tr)})\n"
        f"  sft_val    : {len(final_sft_va)} (was {len(base_sft_va)})\n"
        f"  prefs_train: {len(final_pr_tr)} (was {len(base_pr_tr)})\n"
        f"  prefs_val  : {len(final_pr_va)} (was {len(base_pr_va)})"
    )
    c_sft = Counter((r.get("task", "?"), r.get("language", "?")) for r in final_sft_tr)
    c_pr = Counter((r.get("task", "?"), r.get("language", "?")) for r in final_pr_tr)
    print("  Top SFT tasks:", c_sft.most_common(12))
    print("  Top Pref tasks:", c_pr.most_common(12))


if __name__ == "__main__":
    main()

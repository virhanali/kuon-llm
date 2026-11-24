"""CPU-only probe: can Kuon choose an action and metric from confirmed facts?"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from kuon.chat import format_prompt
from kuon_agent.llm import KuonEngine
from scripts.eval_action_choice import completion_mean_logprob


# Fresh diagnostic cases. Do not reuse these for training.
# Issue, evidence A, evidence B, action+metric A, action+metric B, missing question.
CASES = {
    "id": (
        ("Pesanan sering terlambat dikirim.",
         "Sebagian besar pesanan terlambat karena menunggu dikemas. Kurir mengambil pesanan yang siap tepat waktu.",
         "Sebagian besar pesanan dikemas tepat waktu. Pesanan terlambat karena menunggu kurir mengambilnya.",
         "Uji tambahan petugas kemas pada jam ramai. Ukur waktu tunggu sebelum pengemasan.",
         "Uji jadwal pengambilan tambahan dengan kurir. Ukur waktu tunggu setelah paket siap.",
         "Tanyakan apakah pesanan tertahan sebelum dikemas atau setelah siap diambil."),
        ("Tim dukungan menerima banyak kontak berulang.",
         "Kontak berulang terutama soal tautan pengaturan ulang kata sandi yang kedaluwarsa. Pelanggan bisa melacak kiriman.",
         "Kontak berulang terutama karena pelanggan tidak bisa menemukan tautan pelacakan. Pengaturan ulang kata sandi berjalan.",
         "Uji penerbitan tautan pengaturan ulang baru. Ukur persentase pengguna yang berhasil mengganti kata sandi.",
         "Uji menampilkan tautan pelacakan di konfirmasi pesanan. Ukur jumlah kontak ulang tentang pelacakan.",
         "Tanyakan masalah apa yang paling banyak muncul pada kontak berulang."),
        ("Jumlah pelanggan yang memperpanjang paket servis turun.",
         "Banyak email pengingat gagal terkirim. Pelanggan yang menerima email bisa membayar seperti biasa.",
         "Email pengingat sampai. Pelanggan yang membuka tautan sering mendapat pesan gagal saat membayar.",
         "Uji mengirim pengingat ke alamat email yang sudah dikonfirmasi. Ukur persentase email yang berhasil terkirim.",
         "Uji perbaikan halaman pembayaran. Ukur persentase pelanggan yang berhasil menyelesaikan pembayaran.",
         "Tanyakan apakah pelanggan menerima email pengingat dan apakah pembayaran berhasil."),
    ),
    "en": (
        ("Repair appointments often start late.",
         "Most delayed repairs are waiting for a diagnostic bench. Parts are already available.",
         "Most delayed repairs are diagnosed on time. They are waiting for parts to arrive.",
         "Trial an extra diagnostic bench during peak hours. Measure waiting time before diagnosis.",
         "Trial staging common parts before appointments. Measure waiting time for parts after diagnosis.",
         "Ask whether repairs are waiting for diagnosis or for parts."),
        ("Subscribers contact support repeatedly.",
         "Most repeat contacts are about download links expiring. Billing receipts arrive as expected.",
         "Most repeat contacts are about missing billing receipts. Download links work.",
         "Trial renewing the expired download link on request. Measure the rate of successful downloads.",
         "Trial an automatic receipt resend. Measure repeat contacts about missing receipts.",
         "Ask what issue most repeat contacts are about."),
        ("Fewer people order the lunch special.",
         "Fewer people open the lunch menu. The share of menu visitors who order has not changed.",
         "Just as many people open the lunch menu. More visitors leave the checkout form without ordering.",
         "Trial a prominent link to the lunch menu. Measure visits to the lunch menu.",
         "Trial a shorter checkout form. Measure the share of checkout visitors who place an order.",
         "Ask whether fewer people open the menu or whether more visitors leave checkout."),
    ),
}
COPY = {
    "id": ("Fakta terkonfirmasi", "Pilih satu tindakan dan ukuran yang cocok. Jika fakta kurang, pilih pertanyaan.",
           "Belum diketahui di bagian mana orang atau pekerjaan tertahan.",
           ("Laporan dibuat pada hari Senin.", "Laporan dibuat pada hari Kamis.")),
    "en": ("Confirmed facts", "Choose one action with its metric. If a fact is missing, choose the question.",
           "It is not yet known where people or work are getting held up.",
           ("The report was written on Monday.", "The report was written on Thursday.")),
}


@torch.inference_mode()
def run(engine):
    summary = {}
    letters = "ABC"
    letter_tokens = [engine.tokenizer.encode(" " + letter) for letter in letters]
    assert all(len(ids) == 1 for ids in letter_tokens)
    letter_ids = [ids[0] for ids in letter_tokens]
    for lang, scenarios in CASES.items():
        header, instruction, missing, dates = COPY[lang]
        counts = Counter()
        for i, (issue, a, b, action_a, action_b, ask) in enumerate(scenarios):
            choices = (action_a, action_b, ask)
            facts = (("a", a, dates[0], 0), ("b", b, dates[0], 1),
                     ("irrelevant", a, dates[1], 0), ("missing", missing, dates[0], 2))
            by_case = {}
            by_text = {}
            for kind, confirmed, note, expected in facts:
                body = f"{issue}\n{header}: {confirmed} {note}\n{instruction}"
                guesses = []
                for offset in range(3):
                    order = [(offset + k) % 3 for k in range(3)]
                    options = "\n".join(f"{letter}. {choices[idx]}" for letter, idx in zip(letters, order))
                    prompt = format_prompt(f"{body}\n{options}")
                    ids = engine.tokenizer.encode(prompt)
                    assert all(engine.tokenizer.encode(f"{prompt} {letter}") == ids + tokens
                               for letter, tokens in zip(letters, letter_tokens))
                    logits, _ = engine.model(torch.tensor([ids]))
                    scores = logits[0, -1, letter_ids]
                    guesses.append(order[int(scores.argmax())])
                correct = sum(guess == expected for guess in guesses)
                counts["presentations_correct"] += correct
                counts["presentations_total"] += len(guesses)
                counts[f"{kind}_stable_correct"] += correct == 3
                by_case[kind] = correct == 3
                prompt = format_prompt(body)
                text_scores = [completion_mean_logprob(engine.model, engine.tokenizer, prompt, choice, "cpu")
                               for choice in choices]
                text_best = max(range(3), key=text_scores.__getitem__)
                by_text[kind] = text_best == expected
                counts["text_items_correct"] += by_text[kind]
                counts["text_items_total"] += 1
                print(json.dumps(dict(language=lang, scenario=i, case=kind, issue=issue,
                                      evidence=confirmed, irrelevant_note=note, expected=choices[expected],
                                      selected=[choices[g] for g in guesses], stable_correct=correct == 3,
                                      best_text=choices[text_best], text_correct=by_text[kind]),
                                 ensure_ascii=False))
            counts["relevant_pairs_correct"] += by_case["a"] and by_case["b"]
            counts["unchanged_action_correct"] += by_case["a"] and by_case["irrelevant"]
            counts["fully_correct"] += all(by_case.values())
            counts["text_relevant_pairs_correct"] += by_text["a"] and by_text["b"]
            counts["text_fully_correct"] += all(by_text.values())
            counts["scenarios"] += 1
        summary[lang] = dict(counts)
    print(json.dumps(dict(summary=summary)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    run(KuonEngine(str(args.checkpoint), device="cpu", temperature=0))

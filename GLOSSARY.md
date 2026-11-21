# Kuon — Glosarium

Kuon adalah LLM kecil berbahasa Inggris dengan bobot terbuka, dibangun dari
nol: setiap istilah di bawah ini diimplementasikan langsung di repo ini,
bukan hasil impor dari framework. Istilah dipertahankan dalam bahasa
Inggris (supaya kamu langsung mengenalinya di kode), definisinya dalam
Bahasa Indonesia.

## Bahasa

### Dari teks ke angka

**Token**:
Unit teks yang dibaca dan ditulis model — sebuah angka id di
[0, vocab_size). Bisa berupa kata umum, potongan kata, tanda baca, atau
satu byte mentah.
_Hindari_: kata, karakter

**Tokenizer**:
Kamus dua arah antara teks dan token id (`kuon/tokenizer.py`).
Contoh: `"Once upon a time"` → `[426, 444, 258, 394]`.
_Hindari_: encoder, vocab

**BPE (byte-pair encoding)**:
Cara MEMBUAT tokenizer: mulai dari 256 token byte mentah, lalu berulang
kali gabungkan pasangan dua token yang paling sering muncul bersebelahan
menjadi satu token baru, sampai vocab_size tercapai.

**Byte-level**:
Fondasi tokenizer mencakup semua 256 byte, sehingga teks APA PUN —
emoji, bahasa apa pun — selalu bisa di-encode dan di-decode utuh.

**Special token**:
Penanda kontrol yang tidak dipelajari dari teks. Kuon punya satu:
`<endoftext>` (id 8191) — pemisah antar dokumen di dataset.

**Dokumen**:
Satu unit korpus latihan (satu cerita TinyStories). Semua dokumen
ditumpuk berurutan, dipisah `<endoftext>`.
_Hindari_: baris, record, contoh

**Packed dataset**:
File `.bin`: token id seluruh dokumen digabung jadi satu larik uint16
raksasa, dibaca via memmap saat training (tidak perlu termuat penuh di
RAM).
_Hindari_: shards

### Model

**Context window (`block_size`)**:
Jendela maksimal token yang bisa dilihat model dalam satu kali forward.
Kuon-13m: 512 token.
_Hindari_: block, sequence length

**Block**:
Satu lapisan transformer: attention + feed-forward SwiGLU + dua RMSNorm.
Tidak ada hubungannya dengan block_size.
_Hindari_: layer

**Parameter / weights**:
Angka-angka yang "dipelajari" model. Kuon-13m = 12,59 juta angka.
Sebelum dilatih: acak. Sesudahnya: bermakna.

**Embedding**:
Tabel pencarian dari token id ke vektor `n_embd` angka (384 di Kuon-13m).
Di sinilah "makna" sebuah token hidup.

**Attention**:
Mekanisme "menoleh": tiap token menghitung seberapa penting token-token
sebelumnya baginya, lalu menyerap informasi dari yang penting.

**Causality (causal masking)**:
Aturan main: menebak token ke-t hanya boleh memakai token sebelum t.
Dipaksakan lewat masking; diverifikasi lewat test.

**RoPE**:
Rotary position embedding — cara memberi tahu model POSISI token:
vektor q dan k "diputar" sesuai posisinya, sehingga skor attention hanya
bergantung pada jarak antar token, bukan posisi absolut.

**RMSNorm**:
Penjaga skala angka antar lapisan agar pelatihan stabil. Versi lebih
sederhana dan lebih cepat dari LayerNorm.

**SwiGLU**:
"Otak berpikir" tiap lapisan setelah attention:
`down(silu(gate(x)) * up(x))` — gerbang memutuskan informasi mana yang
lolos.

**GQA**:
Grouped-query attention — kepala key/value lebih sedikit daripada
kepala query (di Kuon-13m: 2 vs 6). Kualitas attention tetap, memori
inferensi jauh lebih kecil.

**Tied embeddings**:
Matriks embedding dipakai ulang sebagai kepala output (lm_head) —
hemat vocab_size × n_embd parameter, gratis tanpa penalti kualitas.

**Looped model (recurrent depth)**:
Blok transformer yang sama dijalankan beberapa kali per token —
komputasi/kedalaman bertambah tanpa menambah parameter. Kuon memakai
`loop_start` / `loop_end` / `loop_passes`; loop blok tengah lebih
menguntungkan daripada blok akhir (temuan Inkling 2 Mini).
_Hindari_: recurrent model (itu RNN)


**KV cache**:
Tempat menyimpan hasil hitung key/value untuk posisi-posisi yang sudah
diproses, supaya tiap token baru saat generation cukup satu forward
kecil — bukan menghitung ulang semuanya.

### Training

**Next-token prediction**:
Satu-satunya tugas pretraining: "lihat 511 token, tebak token ke-512".
Miliaran tebakan → model akhirnya paham bahasa.

**Loss (cross-entropy)**:
Angka kesalahan yang diminimalkan training. Titik nol pembanding:
model acak memberi loss = ln(vocab_size) ≈ 9.02 untuk vocab 8192.
_Hindari_: error

**Val loss**:
Loss yang sama, diukur pada dokumen yang tidak pernah dilatih — ukuran
kejujuran generalisasi. Train loss turun tapi val naik = model menghafal.
_Hindari_: test loss

**Step**:
Satu update optimizer atas block_size × batch_size × grad_accum token
(16.384 token per step di Kuon-13m).
_Hindari_: epoch, iterasi

**Gradient & backpropagation**:
Arah perbaikan: "kalau parameter ini digeser sedikit ke sana, loss turun".
Backprop menghitung arah semua parameter sekaligus, dari belakang ke depan.

**Learning rate / AdamW / cosine + warmup**:
Ukuran langkah tiap update. AdamW: optimizer standar industri. Warmup:
langkah kecil-kecil di awal (pemanasan). Cosine: langkah mengecil halus
menuju akhir training.

**Batch & gradient accumulation**:
Batch = jumlah contoh per update. Kalau batch besar tak muat RAM: hitung
beberapa micro-batch berturut-turut, baru lakukan satu update
(Kuon: 4 micro-batch × 8).

**Checkpoint**:
Potret menyeluruh (weights + optimizer + step) yang bisa dilanjutkan
(resume) atau dimuat untuk generation. Ada `-last.pt` dan `-best.pt`.

**tok/s**:
Kecepatan training: token per detik yang mampu diproses hardware.

**Pretraining**:
Tahap belajar bahasa dari korpus mentah — pekerjaan `kuon/trainer.py`.
_Hindari_: training (terlalu luas)

**SFT / DPO**:
Tahap lanjutan roadmap: SFT = belajar format instruksi/percakapan;
DPO = belajar preferensi (mana jawaban yang lebih disukai manusia).

**Chat template**:
Format teks yang membungkus dialog sebelum masuk model. Kuon memakai
penanda teks biasa: `User: ...\nAssistant: ...` (belum ada special token
khusus chat). Saat SFT, loss hanya dihitung pada bagian `Assistant:`.


### Inference

**Generation (autoregressive)**:
Loop: tebak 1 token → tempel ke teks → tebak lagi → berhenti di
`<endoftext>` atau saat budget token habis.
_Hindari_: decoding (sama artinya tapi lebih kabur)

**Sampling**:
Cara memilih token berikutnya dari distribusi peluang model, dikendalikan
oleh temperature, top-k, dan top-p.
_Hindari_: decoding, greedy (itu satu metode sampling)

**Temperature**:
Seberapa berani model memilih: 0 = selalu token terpopuler (kaku,
deterministik), 0.8 = seimbang, 1.2+ = liar dan kreatif.

**top-k / top-p**:
Saringan keselamatan: hanya boleh memilih dari k token teratas (top-k)
atau dari kumpulan token terkecil yang total peluangnya melampaui p
(top-p / nucleus). Mencegah token aneh bermunculan.

### Deployment

**GGUF / llama.cpp**:
Format file dan runtime untuk menjalankan model di CPU biasa tanpa
Python — target ekspor Kuon agar bisa dipakai di mana saja.

**Quantization**:
Kompresi weights (misal 16-bit → 4-bit) supaya model kecil jalan cepat
dan ringan di laptop.

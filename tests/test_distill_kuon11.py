"""Offline tests for the Kuon 1.1 non-coding Gemma 4 distillation script."""

from scripts.distill_kuon11 import (
    build_casual_identity_rows,
    build_prompt_specs,
    convert_record,
    is_clean_noncoding,
)


def test_noncoding_and_teacher_leak_filters():
    forbidden = {"berapa laba bersih cabang surabaya"}
    assert is_clean_noncoding("Ringkas poin utama dokumen ini dengan jelas.", forbidden)
    assert not is_clean_noncoding("Model ini berbasis Gemma dari Google.", forbidden)
    assert not is_clean_noncoding("Berikut contoh:\n```go\npackage main\n```", forbidden)
    assert not is_clean_noncoding("Gunakan func main() untuk mulai.", forbidden)
    assert not is_clean_noncoding("Berapa laba bersih cabang Surabaya", forbidden)


def test_prompt_specs_cover_all_6_noncoding_pillars_with_id_majority():
    specs = build_prompt_specs(
        open_consult=20, doc_summary=20, writing=20, explain=20, multiturn=20, casual=20
    )
    assert len(specs) == 120
    kinds = {s["kind"] for s in specs}
    assert kinds == {"casual", "open_consult", "doc_summary", "writing", "explain", "multiturn"}
    id_ratio = sum(1 for s in specs if s["lang"] == "id") / len(specs)
    assert 0.60 <= id_ratio <= 0.70


def test_convert_record_doc_summary_grounding_and_coding_rejection():
    sft: list[dict] = []
    prefs: list[dict] = []
    spec = {"id": "doc-0", "kind": "doc_summary", "lang": "id"}

    good_obj = {
        "title": "Tech Spec Caching Artikel",
        "doc": "Latensi endpoint artikel saat ini 420 ms pada 120 RPS. Redis cache dengan TTL 300 detik menargetkan p95 di bawah 80 ms.",
        "sum_q": "Ringkas poin utama dan angka penting dari dokumen ini.",
        "sum_a": "Berdasarkan [1], latensi saat ini 420 ms pada 120 RPS dan diusulkan Redis cache TTL 300 detik agar p95 di bawah 80 ms.",
        "qa_q": "Berapa TTL Redis yang diusulkan di dokumen ini?",
        "qa_a": "Menurut [1], TTL Redis yang diusulkan adalah 300 detik.",
        "unans_q": "Berapa biaya bulanan server Redis yang dianggarkan?",
        "unans_a": "Dokumen [1] tidak menyebutkan anggaran biaya bulanan server Redis. Bisa bagikan rincian biayanya?",
    }
    assert convert_record(spec, good_obj, set(), sft, prefs)
    assert len(sft) == 3
    assert len(prefs) == 2

    # Hallucinated number (999 ms) must be rejected
    bad_num = dict(good_obj, sum_a="Menurut [1], latensi mencapai 999 ms.")
    assert not convert_record(spec, bad_num, set(), sft, prefs)

    # Code snippet inside writing must be rejected
    write_spec = {"id": "w-0", "kind": "writing", "lang": "id"}
    assert not convert_record(
        write_spec,
        {"user": "Buatkan contoh balasan", "assistant": "```python\nprint('hi')\n```"},
        set(),
        sft,
        prefs,
    )


def test_curated_casual_identity_rows_have_zero_leaks():
    rows = build_casual_identity_rows()
    assert len(rows) >= 60
    assert all(is_clean_noncoding(r["assistant"], set()) for r in rows)

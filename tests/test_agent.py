"""Tests for the from-scratch agent: BM25, chunking, and the loop (offline)."""

from kuon_agent.agent import AgentConfig, KuonAgent
from kuon_agent.rag import BM25Index, chunk_text
from kuon_agent.search import Source


def make_index(vocab_source: str = "test") -> BM25Index:
    index = BM25Index()
    index.add_text(
        "MrBeast is an American YouTuber known for elaborate challenges and philanthropy.",
        "wiki:MrBeast",
    )
    index.add_text("The Eiffel Tower is a wrought-iron tower in Paris, France.", "wiki:Eiffel")
    index.add_text("TinyStories is a synthetic dataset of short children's stories.", "wiki:TinyStories")
    return index.finalize()


def test_bm25_retrieves_the_right_chunk():
    hits = make_index().search("who is mrbeast", top_k=2)
    assert hits, "expected at least one hit"
    assert "MrBeast" in hits[0][1].text


def test_chunking_splits_long_text():
    text = "short paragraph.\n\n" + ("word " * 400)
    chunks = chunk_text(text, "doc", max_chars=300, overlap_chars=50)
    assert len(chunks) >= 2
    assert all(c.source == "doc" for c in chunks)


class RecordingEngine:
    name = "recording"

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        if "Bullet points:" in prompt:
            return "- MrBeast is a YouTuber [1]"
        return "MrBeast is a YouTuber known for challenges [1]."


def test_agent_loop_grounds_answer_in_evidence():
    engine = RecordingEngine()
    agent = KuonAgent(engine, AgentConfig(use_wiki=False, use_web=False),
                      rag_index=make_index())
    result = agent.answer("who is mrbeast")

    assert "MrBeast" in result.answer
    assert result.sources and result.sources[0]["title"] == "wiki:MrBeast"
    assert result.notes  # think stage ran
    # evidence (with the retrieved text) must reach the final prompt
    assert "elaborate challenges" in engine.prompts[-1]


def test_agent_admits_ignorance_without_evidence():
    engine = RecordingEngine()
    agent = KuonAgent(engine, AgentConfig(use_wiki=False, use_web=False), rag_index=None)
    result = agent.answer("who is mrbeast")
    assert "don't know" in result.answer.lower()
    assert result.sources == []
    assert not engine.prompts  # no model call without evidence


def test_agent_uses_injected_search_functions():
    engine = RecordingEngine()
    agent = KuonAgent(
        engine,
        AgentConfig(use_wiki=True, use_web=False, think=False),
        rag_index=None,
        wiki_fn=lambda q, n: [Source(title="Injected", url="https://example.org", text="Injected evidence.")],
        web_fn=lambda q, n: [],
    )
    result = agent.answer("anything")
    assert result.sources[0]["title"] == "Injected"
    assert "Injected evidence." in engine.prompts[-1]


def test_smalltalk_answers_without_search():
    engine = RecordingEngine()
    agent = KuonAgent(engine, AgentConfig(use_wiki=True, use_web=False), rag_index=make_index(),
                      wiki_fn=lambda q, n: [Source(title="Wiki", url="", text="Halo the game.")])
    result = agent.answer("halo")
    assert result.sources == []
    assert engine.prompts[-1] == "User: halo\nAssistant:"

    story = agent.answer("Tell me a short story about a cat.")
    assert story.sources == []

    elongated = agent.answer("haloooo")
    assert elongated.sources == []


def test_identity_variants_skip_web_search_and_canonicalize():
    engine = RecordingEngine()
    web_called: list[str] = []
    agent = KuonAgent(
        engine,
        AgentConfig(use_wiki=True, use_web=True),
        rag_index=make_index(),
        wiki_fn=lambda q, n: [Source(title="Wiki", url="", text="Should not run")],
        web_fn=lambda q, n: (web_called.append(q) or [Source(title="Web", url="https://x.test", text="Hy4")]),
    )
    for q in [
        "model llm apa kamu?",
        "model apa kamu?",
        "kamu model apa?",
        "kamu pakai gpt?",
        "siapa kamu?",
        "kamu siapa?",
        "kamu didevelop sama siapa?",
        "kamu dibuat oleh siapa?",
        "Halo, model llm apa kamu?",
    ]:
        res = agent.answer(q)
        assert res.sources == []
        assert engine.prompts[-1] == "User: Kamu dasarnya model apa?\nAssistant:"

    for q in [
        "what model are you?",
        "what llm are you?",
        "are you based on gpt?",
        "who are you?",
    ]:
        res = agent.answer(q)
        assert res.sources == []
        assert engine.prompts[-1] == "User: What base model are you built on?\nAssistant:"

    assert web_called == []
    # General conceptual LLM question must still search normally, not trigger identity shortcut
    res_general = agent.answer("Apa itu LLM?")
    assert web_called == ["Apa itu LLM?"]
    assert len(res_general.sources) >= 1


def test_clean_source_label_strips_internal_paths():
    from kuon_agent.agent import _clean_source_label

    assert _clean_source_label("/data/home/kuon-llm/kb/kuon_studio.md") == "kuon_studio.md"
    assert _clean_source_label("https://example.org/doc") == "https://example.org/doc"
    assert _clean_source_label("tool:table_analytics") == "tool:table_analytics"


def test_attached_markdown_doc_summarizes_and_answers_without_kb_pollution():
    class BudgetingRecordingEngine(RecordingEngine):
        def prompt_fits(self, prompt: str, max_tokens: int = 128) -> bool:
            return len(prompt) < 4000

    engine = BudgetingRecordingEngine()
    agent = KuonAgent(engine, AgentConfig(use_wiki=False, use_web=False), rag_index=make_index())
    doc = (
        "# Tech Spec: Caching API Article\n"
        "## Ringkasan\n"
        "Endpoint GET /api/v1/articles memiliki latensi 420 ms saat 120 RPS.\n"
        "## Solusi\n"
        "Tambahkan Redis cache dengan TTL 300 detik untuk menurunkan p95 di bawah 80 ms."
    )
    attached_turn = f"[Attachment: 20261119 - Tech Spec - Caching API Article.md]\n{doc}\n[/Attachment]\nRingkas poin utama dan angka penting dari dokumen ini."
    res_sum = agent.advise([("user", attached_turn)])
    assert "Caching API Article" in res_sum.answer
    assert "420 ms" in res_sum.answer or "300 detik" in res_sum.answer
    assert "document_extract" in res_sum.notes
    assert "MrBeast" not in res_sum.answer

    # Specific follow-up question in turn 2 must ground ONLY on the uploaded file, not make_index()
    res_q = agent.advise([
        ("user", attached_turn),
        ("assistant", res_sum.answer),
        ("user", "Apa solusi caching dan TTL Redis di dokumen ini?"),
    ])
    assert "Redis cache" in engine.prompts[-1]
    assert "MrBeast" not in engine.prompts[-1]
    assert res_q.sources and res_q.sources[0]["title"] == "20261119 - Tech Spec - Caching API Article.md"


def test_capability_open_symptom_and_multiturn_table_to_kpi_transition():
    class BudgetingRecordingEngine(RecordingEngine):
        def prompt_fits(self, prompt: str, max_tokens: int = 128) -> bool:
            return len(prompt) < 4000

    engine = BudgetingRecordingEngine()
    web_called: list[str] = []
    agent = KuonAgent(
        engine,
        AgentConfig(use_wiki=True, use_web=True),
        rag_index=make_index(),
        web_fn=lambda q, n: (web_called.append(q) or [Source(title="Web", url="https://x.test", text="Soft skill")]),
    )

    # 1. Capability question skips web search
    cap = agent.answer("skill apa aja yang kamu punya?")
    assert web_called == []
    assert "Kuon 1" in cap.answer
    assert "KPI" in cap.answer

    # 2. Open business symptom asks concrete diagnostic numbers instead of unrelated KB refusal
    sym = agent.advise([("user", "bisnis ku lesu")])
    assert "Omzet 2 periode" in sym.answer

    # 3. Multi-turn: sales change -> new TSV table max -> gross margin KPI must NOT collide
    turn1 = "| Periode | Omzet |\n|---|---|\n| Minggu 1 | Rp25.000.000 |\n| Minggu 2 | Rp20.000.000 |\nBerapa perubahan omzet dari Minggu 1 ke Minggu 2?"
    r1 = agent.advise([("user", turn1)])
    assert "-20%" in r1.answer

    turn2 = "Produk\tOmzet\nKopi Susu\t15000000\nTeh Tarik\t41000000\nRoti Bakar\t10000000\nProduk mana dengan omzet tertinggi dan berapa persen porsinya?"
    r2 = agent.advise([("user", turn1), ("assistant", r1.answer), ("user", turn2)])
    assert "Teh Tarik" in r2.answer and "62,12%" in r2.answer

    turn3 = "Pendapatan bulan ini Rp200.000.000 dan HPP Rp140.000.000. Berapa gross margin bulan ini?"
    r3 = agent.advise([
        ("user", turn1), ("assistant", r1.answer),
        ("user", turn2), ("assistant", r2.answer),
        ("user", turn3),
    ])
    assert "30%" in r3.answer

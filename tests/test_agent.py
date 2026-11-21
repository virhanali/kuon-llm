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

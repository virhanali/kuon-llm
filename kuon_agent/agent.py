"""The Kuon Agent loop: gather evidence -> think -> answer with citations.

Design goals, in order:
1. Grounded: answer ONLY from collected evidence; admit ignorance otherwise.
2. Inspectable: return the evidence, the notes, and the final answer.
3. Swappable: any engine (Kuon, Ollama/Qwen, llama.cpp) works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kuon_agent.rag import BM25Index
from kuon_agent.search import Source, duckduckgo_search, wikipedia_search


@dataclass
class AgentConfig:
    use_rag: bool = True
    use_wiki: bool = True
    use_web: bool = False          # opt-in: DuckDuckGo scrape
    fallback_web: bool = False     # if no evidence, try the web
    think: bool = True             # two-stage: notes -> answer
    top_k_rag: int = 3
    max_sources: int = 4
    max_source_chars: int = 1200
    max_answer_tokens: int = 300


ANSWER_PROMPT = """You are a careful assistant. Answer the question using ONLY the evidence below.
If the evidence does not contain the answer, reply exactly: "I don't know from my sources."
Cite evidence numbers like [1] where you use them. Be concise.

Question: {question}
{notes}Evidence:
{evidence}

Answer:"""

NOTES_PROMPT = """Read the evidence and write 1-3 short bullet points that help answer the question.
Only state what the evidence says. If nothing is relevant, write: no relevant evidence.

Evidence:
{evidence}

Question: {question}

Bullet points:"""


@dataclass
class AgentResult:
    answer: str
    notes: str = ""
    sources: list[dict] = field(default_factory=list)


class KuonAgent:
    def __init__(self, engine, config: AgentConfig | None = None,
                 rag_index: BM25Index | None = None,
                 wiki_fn: Callable[[str, int], list[Source]] = wikipedia_search,
                 web_fn: Callable[[str, int], list[Source]] = duckduckgo_search):
        self.engine = engine
        self.cfg = config or AgentConfig()
        self.rag = rag_index
        self._wiki = wiki_fn
        self._web = web_fn

    # ------------------------------------------------------------ evidence

    def _gather(self, question: str) -> list[Source]:
        gathered: list[Source] = []
        if self.cfg.use_rag and self.rag is not None:
            for _score, chunk in self.rag.search(question, top_k=self.cfg.top_k_rag):
                gathered.append(Source(title=chunk.source, url="", text=chunk.text))
        if self.cfg.use_wiki:
            gathered.extend(self._wiki(question, self.cfg.max_sources))
        if self.cfg.use_web:
            gathered.extend(self._web(question, self.cfg.max_sources))
        if self.cfg.fallback_web and not gathered:
            gathered.extend(self._web(question, self.cfg.max_sources))

        # dedupe by url/title prefix, trim, cap
        seen: set[str] = set()
        unique: list[Source] = []
        for source in gathered:
            key = source.url or source.title[:80]
            if key in seen:
                continue
            seen.add(key)
            unique.append(Source(title=source.title, url=source.url,
                                 text=source.text[:self.cfg.max_source_chars]))
        return unique[:self.cfg.max_sources]

    @staticmethod
    def _format_evidence(sources: list[Source]) -> str:
        blocks = []
        for i, source in enumerate(sources, start=1):
            header = f"[{i}] {source.title}"
            if source.url:
                header += f" ({source.url})"
            blocks.append(f"{header}\n{source.text}")
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- loop

    def answer(self, question: str) -> AgentResult:
        sources = self._gather(question)
        if not sources:
            return AgentResult(
                answer="I don't know from my sources, and no evidence was found.",
                notes="no evidence collected",
                sources=[],
            )

        evidence = self._format_evidence(sources)
        notes = ""
        if self.cfg.think:
            notes = self.engine.complete(
                NOTES_PROMPT.format(evidence=evidence, question=question),
                max_tokens=120,
            ).strip()

        notes_block = f"Notes:\n{notes}\n\n" if notes else ""
        answer = self.engine.complete(
            ANSWER_PROMPT.format(question=question, notes=notes_block, evidence=evidence),
            max_tokens=self.cfg.max_answer_tokens,
        ).strip()

        return AgentResult(
            answer=answer,
            notes=notes,
            sources=[{"title": s.title, "url": s.url} for s in sources],
        )

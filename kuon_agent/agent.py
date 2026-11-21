"""The Kuon Agent loop: gather evidence -> think -> answer with citations.

Design goals, in order:
1. Grounded: answer ONLY from collected evidence; admit ignorance otherwise.
2. Inspectable: return the evidence, the notes, and the final answer.
3. Swappable: any engine (Kuon, Ollama/Qwen, llama.cpp) works unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from kuon.chat import format_prompt
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


ANSWER_PROMPT = """User: {question}
Evidence:
{evidence}
Assistant:"""

NOTES_PROMPT = """Read the evidence and write 1-3 short bullet points that help answer the question.
Only state what the evidence says. If nothing is relevant, write: no relevant evidence.

Evidence:
{evidence}

Question: {question}

Bullet points:"""


SMALLTALK_PATTERNS = [
    re.compile(r"^(halo+|hallo+|hai+|hi+|hello+|hey+|selamat\s+(pagi|siang|sore|malam)|good\s+(morning|afternoon|evening)|assalamualaikum)\b", re.I),
    re.compile(r"\b(thank you|thanks|terima kasih|makasih)\b", re.I),
    re.compile(r"\b(who are you|what is kuon|who is kuon|tell me about yourself|apa itu kuon|siapa kamu)\b", re.I),
    re.compile(r"\b(tell me|write|tulis|ceritakan)\b.{0,40}\b(story|cerita)\b", re.I),
]


def is_smalltalk(question: str) -> bool:
    """Greetings, thanks, and identity questions are answered directly
    (no search) using the chat template the model was fine-tuned on."""
    return any(pattern.search(question.strip()) for pattern in SMALLTALK_PATTERNS)


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
        if is_smalltalk(question):
            # answer in the model's own chat format, without search
            text = self.engine.complete(format_prompt(question),
                                        max_tokens=self.cfg.max_answer_tokens).strip()
            return AgentResult(answer=text, notes="", sources=[])

        sources = self._gather(question)
        if not sources:
            # short, non-question inputs are almost always conversational
            # ("haloooo", "p"); real questions must admit ignorance instead
            looks_factual = bool(re.match(
                r"^(who|what|where|when|why|how|which|siapa|apa|kapan|dimana|di mana|berapa|mengapa|kenapa)\b",
                question.strip(), re.I)) or question.strip().endswith("?")
            if not looks_factual and len(question.split()) <= 3:
                text = self.engine.complete(format_prompt(question),
                                            max_tokens=self.cfg.max_answer_tokens).strip()
                return AgentResult(answer=text, notes="", sources=[])
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

        answer = self.engine.complete(
            ANSWER_PROMPT.format(question=question, evidence=evidence),
            max_tokens=self.cfg.max_answer_tokens,
        ).strip()

        return AgentResult(
            answer=answer,
            notes=notes,
            sources=[{"title": s.title, "url": s.url} for s in sources],
        )

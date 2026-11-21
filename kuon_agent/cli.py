"""CLI: ask the Kuon Agent a question.

Examples:
    python -m kuon_agent.cli "who is mr beast"                 # wiki, mock engine
    python -m kuon_agent.cli "who is mr beast" --engine kuon \
        --checkpoint checkpoints/kuon-13m/kuon-13m-last.pt
    python -m kuon_agent.cli "explain BM25" --engine ollama --model qwen2.5-coder:7b
    python -m kuon_agent.cli "apa itu kuon" --rag notes.txt --no-wiki
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kuon_agent.agent import AgentConfig, KuonAgent
from kuon_agent.llm import KuonEngine, LlamaCppEngine, MockEngine, OllamaEngine
from kuon_agent.rag import BM25Index


def build_engine(args):
    if args.engine == "mock":
        return MockEngine()
    if args.engine == "kuon":
        if not args.checkpoint:
            sys.exit("--engine kuon requires --checkpoint")
        return KuonEngine(args.checkpoint, device=args.device)
    if args.engine == "ollama":
        return OllamaEngine(model=args.model or "qwen2.5-coder:7b", host=args.host)
    if args.engine == "llamacpp":
        if not args.model:
            sys.exit("--engine llamacpp requires --model <file.gguf>")
        return LlamaCppEngine(args.model, binary=args.binary)
    raise ValueError(args.engine)


def build_rag(path: str | None) -> BM25Index | None:
    if not path:
        return None
    index = BM25Index()
    target = Path(path)
    if target.is_dir():
        for file in sorted(target.glob("**/*")):
            if file.suffix.lower() in {".txt", ".md"}:
                index.add_file(str(file), source=str(file))
    else:
        index.add_file(str(target), source=target.name)
    if len(index) == 0:
        sys.exit(f"no text found in {target}")
    return index.finalize()


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the Kuon Agent a question.")
    ap.add_argument("question")
    ap.add_argument("--engine", choices=["mock", "kuon", "ollama", "llamacpp"], default="mock")
    ap.add_argument("--checkpoint", help="Kuon checkpoint for --engine kuon")
    ap.add_argument("--model", help="Ollama model name or GGUF path")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--binary", default="llama-cli")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rag", help="text file or directory indexed with BM25")
    ap.add_argument("--wiki", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--web", action=argparse.BooleanOptionalAction, default=False,
                    help="use DuckDuckGo HTML search (opt-in)")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--show-notes", action="store_true")
    args = ap.parse_args()

    agent = KuonAgent(
        build_engine(args),
        AgentConfig(
            use_rag=bool(args.rag),
            use_wiki=args.wiki,
            use_web=args.web,
            fallback_web=args.web,
            think=args.think,
        ),
        rag_index=build_rag(args.rag),
    )

    result = agent.answer(args.question)
    print(result.answer)
    if args.show_notes and result.notes:
        print("\n--- notes ---")
        print(result.notes)
    if result.sources:
        print("\n--- sources ---")
        for i, source in enumerate(result.sources, start=1):
            line = f"[{i}] {source['title']}"
            if source["url"]:
                line += f" - {source['url']}"
            print(line)


if __name__ == "__main__":
    main()

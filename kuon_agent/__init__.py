"""Kuon Agent: a from-scratch search -> think -> answer loop.

No agent framework. Three small parts:

    rag.py     BM25 retrieval over local text (implemented here)
    search.py  Wikipedia API + DuckDuckGo HTML search (stdlib urllib)
    llm.py     pluggable engines: Kuon checkpoint, Ollama, llama.cpp, mock
    agent.py   the loop: gather evidence -> think -> answer with citations

The model is a swappable engine; the agent is the system around it.
"""

__version__ = "0.1.0"

from kuon_agent.agent import AgentConfig, KuonAgent

__all__ = ["AgentConfig", "KuonAgent", "__version__"]

"""Tests for the OpenAI-compatible server (loopback only, no network deps)."""

import json
import threading
from http.client import HTTPConnection

from kuon_agent.agent import AgentResult
from kuon_agent.server import build_server


class StubAgent:
    engine = type("E", (), {"name": "stub"})()

    def __init__(self, prefix: str = "answer to"):
        self.prefix = prefix

    def answer(self, question: str) -> AgentResult:
        return AgentResult(
            answer=f"{self.prefix}: {question}",
            notes="stub notes",
            sources=[{"title": "Injected source", "url": "https://example.org"}],
        )


def with_server(fn, agents=None):
    default = StubAgent()
    server = build_server(default, host="127.0.0.1", port=0, agents=agents)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        fn(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def test_models_and_health():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/v1/models")
        response = conn.getresponse()
        data = json.loads(response.read())
        assert data["data"][0]["id"] == "kuon-agent"

        conn.request("GET", "/health")
        assert json.loads(conn.getresponse().read())["status"] == "ok"
        conn.close()

    with_server(check)


def test_chat_completion_and_sources():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({"model": "kuon-agent",
                           "messages": [{"role": "user", "content": "who is x"}]})
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read())
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "answer to: who is x"
        assert data["x_sources"][0]["title"] == "Injected source"
        assert data["usage"]["total_tokens"] > 0
        conn.close()

    with_server(check)


def test_model_routing_between_engines():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({"model": "qwen",
                           "messages": [{"role": "user", "content": "hi"}]})
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        assert data["model"] == "qwen"
        assert data["choices"][0]["message"]["content"] == "qwen says: hi"
        conn.close()

    with_server(check, agents={"kuon-agent": StubAgent(), "qwen": StubAgent("qwen says")})


def test_streaming_sse():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json"})
        raw = conn.getresponse().read().decode()
        assert "chat.completion.chunk" in raw
        assert "answer to: hi" in raw
        assert "data: [DONE]" in raw
        conn.close()

    with_server(check)


def test_homepage_serves_chat_ui():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        html = conn.getresponse().read().decode()
        assert "Kuon Studio" in html
        assert "/v1/chat/completions" in html
        conn.close()

    with_server(check)

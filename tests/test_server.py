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


def with_server(fn, agents=None, *, api_key=None):
    default = StubAgent()
    server = build_server(default, host="127.0.0.1", port=0, agents=agents, api_key=api_key)
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
        frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        stop_frame = next(f for f in frames if f["choices"][0]["finish_reason"] == "stop")
        assert stop_frame["x_notes"] == "stub notes"
        assert stop_frame["x_sources"][0]["title"] == "Injected source"
        conn.close()

    with_server(check)


def test_homepage_serves_chat_ui():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        html = conn.getresponse().read().decode()
        assert "Kuon Studio" in html
        assert "/v1/chat/completions" in html
        assert "x_kuon_mode" in html
        assert "x_notes" in html
        assert "stream: true" in html
        conn.close()

    with_server(check)


def test_bearer_api_key_auth():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        # /health stays open for load-balancer/InstaBox probes
        conn.request("GET", "/health")
        r_health = conn.getresponse()
        health = json.loads(r_health.read())
        assert r_health.status == 200 and health["status"] == "ok" and health["busy"] is False

        # /v1/models rejects missing or wrong key
        conn.request("GET", "/v1/models")
        r_unauth = conn.getresponse()
        r_unauth.read()
        assert r_unauth.status == 401

        conn.request("GET", "/v1/models", headers={"Authorization": "Bearer wrong"})
        r_bad = conn.getresponse()
        r_bad.read()
        assert r_bad.status == 401

        # /v1/chat/completions succeeds with valid Bearer key
        body = json.dumps({"messages": [{"role": "user", "content": "halo"}]})
        conn.request(
            "POST", "/v1/chat/completions", body,
            {"Content-Type": "application/json", "Authorization": "Bearer secret-kuon-123"},
        )
        r_ok = conn.getresponse()
        data = json.loads(r_ok.read())
        assert r_ok.status == 200
        assert data["choices"][0]["message"]["content"] == "answer to: halo"
        conn.close()

    with_server(check, api_key="secret-kuon-123")


def test_live_token_streaming_and_params():
    class LiveStreamAgent:
        engine = type("E", (), {"name": "kuon", "quant_mode": "int8"})()

        def __init__(self):
            self.seen_kwargs = None

        def advise(self, history, *, on_token=None, temperature=None, max_tokens=None):
            self.seen_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            parts = ["Halo ", "bisnis ", "Kuon!"]
            if on_token is not None:
                for p in parts:
                    on_token(p)
            return AgentResult(answer="".join(parts), notes="live", sources=[])

    live = LiveStreamAgent()

    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({
            "model": "kuon-agent",
            "x_kuon_mode": "business",
            "stream": True,
            "temperature": 0.15,
            "max_tokens": 96,
            "messages": [{"role": "user", "content": "Bantu cek strategi toko"}],
        })
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        raw = conn.getresponse().read().decode()
        frames = [json.loads(ln[6:]) for ln in raw.splitlines() if ln.startswith("data: {")]
        deltas = [f["choices"][0]["delta"].get("content") for f in frames if "content" in f["choices"][0]["delta"]]
        assert deltas == ["Halo ", "bisnis ", "Kuon!"]
        assert live.seen_kwargs == {"temperature": 0.15, "max_tokens": 96}
        conn.close()

    with_server(check, agents={"kuon-agent": live})


def test_tool_stream_guard_and_tsv_excel_table():
    from kuon_agent.agent import AgentConfig, KuonAgent

    class StreamToolEngine:
        name = "stream-stub"

        def __init__(self):
            self.calls = 0

        def prompt_fits(self, prompt: str, max_tokens: int) -> bool:
            return True

        def complete_stream(self, prompt: str, max_tokens: int = 256, *, temperature=None):
            self.calls += 1
            if self.calls == 1:
                for tok in ("TO", "OL:", " calc(10 + 5)"):
                    yield tok
            else:
                for tok in ("Hasilnya ", "15."):
                    yield tok

        def complete(self, prompt: str, max_tokens: int = 256, *, temperature=None) -> str:
            return "".join(self.complete_stream(prompt, max_tokens=max_tokens, temperature=temperature))

    agent = KuonAgent(StreamToolEngine(), AgentConfig(use_rag=False, use_wiki=False, use_web=False))
    emitted: list[str] = []
    res = agent.advise([("user", "Hitung 10 ditambah 5")], on_token=emitted.append)
    assert res.answer == "Hasilnya 15."
    # TOOL: syntax must NEVER leak into live streamed tokens
    assert all("TOOL:" not in chunk for chunk in emitted)
    assert "".join(emitted) == "Hasilnya 15."

    # Excel/Google Sheets tab-separated table with "Omset" header works deterministically
    tsv_msg = "Periode\tOmset\tPesanan\nM1\tRp50.000.000\t400\nM2\tRp40.000.000\t320\nBerapa persen perubahan omset dari M1 ke M2?"
    res_tsv = agent.advise([("user", tsv_msg)])
    assert "-20%" in res_tsv.answer
    assert "pct_change" in res_tsv.notes


def test_server_validation_and_query_string():
    def check(port: int):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        # Health with query string still returns 200
        conn.request("GET", "/health?probe=1")
        r = conn.getresponse()
        r.read()
        assert r.status == 200

        # Invalid JSON returns 400 (not 500)
        conn.request("POST", "/v1/chat/completions", "{bad json", {"Content-Type": "application/json"})
        r_bad = conn.getresponse()
        r_bad.read()
        assert r_bad.status == 400

        # Invalid temperature returns 400
        bad_temp = json.dumps({"messages": [{"role": "user", "content": "hi"}], "temperature": "hot"})
        conn.request("POST", "/v1/chat/completions", bad_temp, {"Content-Type": "application/json"})
        r_temp = conn.getresponse()
        r_temp.read()
        assert r_temp.status == 400

        # Oversized Content-Length returns 413 without reading body into RAM
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(2_000_000))
        conn.endheaders()
        r_big = conn.getresponse()
        r_big.read()
        assert r_big.status == 413
        conn.close()

    with_server(check)


def test_identity_routing_indonesian_fallback_rag_and_distill_guards():
    from kuon_agent.agent import AgentConfig, KuonAgent, is_smalltalk
    from kuon_agent.llm import MockEngine
    from kuon_agent.rag import chunk_text
    from scripts.distill_gemma4 import build_identity_sft_rows, convert_evidence_obj, is_clean_text

    # Base-model / creator questions route directly to Kuon (no Wikipedia search)
    assert is_smalltalk("kamu dasarnya model apa?")
    assert is_smalltalk("What base model are you built on?")
    assert is_smalltalk("Siapa yang membuatmu?")
    from kuon_agent.agent import _normalize_smalltalk_prompt
    assert _normalize_smalltalk_prompt("Halo, kamu dasarnya model apa?") == "kamu dasarnya model apa?"
    assert _normalize_smalltalk_prompt("Halo") == "Halo"

    # Indonesian factual query with no sources replies in Indonesian (not English)
    agent = KuonAgent(MockEngine(), AgentConfig(use_rag=False, use_wiki=False, use_web=False))
    res_id = agent.answer("Berapa omzet cabang Mars bulan lalu?")
    assert "tidak" in res_id.answer.lower() or "belum" in res_id.answer.lower()

    # Oversized single-paragraph document splits cleanly into <= max_chars chunks
    chunks = chunk_text("A" * 2500, source="big.txt", max_chars=900, overlap_chars=120)
    assert len(chunks) >= 3 and all(len(c.text) <= 900 for c in chunks)

    # Distillation guards catch third-party brands and handle multi-dot Rupiah amounts
    assert not is_clean_text("Saya berbasis Claude dari Anthropic.", set())
    assert not is_clean_text("Powered by Gemma from Google.", set())
    id_rows = build_identity_sft_rows()
    assert any("dasarnya model apa" in r["user"].lower() for r in id_rows)

    sft_out, prefs_out = [], []
    ok = convert_evidence_obj(
        "id",
        {
            "evidence": "[1] Rekap Omzet\nCabang A: Rp50.000.000. Cabang B: Rp12.500.000.",
            "ans_q": "Berapa omzet Cabang B?",
            "ans_a": "Menurut rekap [1], omzet Cabang B sebesar Rp12.500.000.",
            "ans_wrong": "Menurut rekap [1], omzet Cabang B sebesar Rp50.000.000.",
            "unans_q": "Berapa laba bersih Cabang B?",
            "unans_a": "Laba bersih tidak disebutkan dalam kutipan [1]. Bisa bagikan biaya operasionalnya?",
            "unans_wrong": "Laba bersih Cabang B adalah Rp50.000.000.",
        },
        set(), sft_out, prefs_out,
    )
    assert ok and len(sft_out) == 2 and len(prefs_out) == 3


def test_hybrid_rag_rrf_and_fallback():
    import numpy as np
    from kuon_agent.rag import Chunk, HybridIndex

    class ToyEmbedder:
        def _vec(self, text: str) -> np.ndarray:
            t = text.lower()
            v = np.array([
                1.0 if any(w in t for w in ("churn", "kabur", "berhenti")) else 0.0,
                1.0 if any(w in t for w in ("laba", "profit", "margin")) else 0.0,
                1.0 if any(w in t for w in ("inv-2094", "retur", "paket")) else 0.0,
            ], dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

        def embed_passages(self, texts: list[str]):
            return np.vstack([self._vec(t) for t in texts])

        def embed_query(self, query: str):
            return self._vec(query)

    idx = HybridIndex(embedder=ToyEmbedder(), min_semantic_score=0.5)
    idx.add(Chunk("Tingkat churn pelanggan berlangganan naik dari 3,1% menjadi 7,8%.", "retensi.md"))
    idx.add(Chunk("Laba bersih cabang Bandung setelah pajak mencapai Rp42 juta.", "keuangan.md"))
    idx.add(Chunk("Pesanan #INV-2094 diretur karena alamat penerima tidak ditemukan.", "logistik.md"))
    idx.finalize()

    # Zero lexical overlap ("mengapa pelanggan kabur?") still retrieves churn doc via semantic vector
    hits_sem = idx.search("mengapa pelanggan kabur?", top_k=1)
    assert hits_sem and hits_sem[0][1].source == "retensi.md"

    # Cross-lingual EN -> ID ("net profit") retrieves Indonesian keuangan doc
    hits_xl = idx.search("What is the net profit?", top_k=1)
    assert hits_xl and hits_xl[0][1].source == "keuangan.md"

    # Exact invoice ID still ranks #1 via BM25 + semantic RRF
    hits_id = idx.search("Status #INV-2094?", top_k=1)
    assert hits_id and hits_id[0][1].source == "logistik.md"

    # Graceful fallback to pure BM25 when embedder=None
    bm_only = HybridIndex(embedder=None)
    bm_only.add(Chunk("Pesanan #INV-2094 diretur karena alamat tidak ditemukan.", "logistik.md"))
    hits_fb = bm_only.search("INV-2094", top_k=1)
    assert hits_fb and hits_fb[0][1].source == "logistik.md"


def test_exa_search_trust_diversity_and_fallback(monkeypatch):
    import io
    from kuon_agent import search as search_mod

    # Without EXA_API_KEY, exa_search degrades cleanly to []
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert search_mod.exa_search("presiden indonesia sekarang") == []

    captured: dict = {}

    class FakeResp:
        def __init__(self, data: bytes):
            self._buf = io.BytesIO(data)

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_payload = {
        "results": [
            {
                "title": "Blog Pribadi Politik",
                "url": "https://blog-acak.example/post-1",
                "publishedDate": "2026-01-10T00:00:00.000Z",
                "highlights": ["Opini blog tanpa sumber resmi."],
            },
            {
                "title": "Fake Reuters Lookalike",
                "url": "https://fakereuters.com/news",
                "publishedDate": "2026-02-01T00:00:00.000Z",
                "highlights": ["Domain tiruan tidak boleh masuk prioritas tepercaya."],
            },
            {
                "title": "Sekretariat Negara RI",
                "url": "https://www.setneg.go.id/presiden-ri",
                "publishedDate": "2025-10-20T08:00:00.000Z",
                "highlights": ["Prabowo Subianto menjabat sebagai Presiden Republik Indonesia sejak 20 Oktober 2024."],
            },
            {
                "title": "Artikel Setneg Duplikat",
                "url": "https://setneg.go.id/profil-lain",
                "publishedDate": "2025-10-21T08:00:00.000Z",
                "highlights": ["Seharusnya di-dedup karena domain setneg.go.id sudah ada."],
            },
            {
                "title": "ANTARA News — Presiden RI",
                "url": "https://www.antaranews.com/berita/presiden",
                "publishedDate": "2026-03-05T12:00:00.000Z",
                "highlights": ["Presiden Prabowo Subianto memimpin rapat kabinet di Jakarta."],
            },
        ]
    }

    def fake_urlopen(req, timeout=12.0):
        captured["url"] = req.full_url
        captured["key"] = req.get_header("X-api-key")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp(json.dumps(fake_payload).encode("utf-8"))

    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("EXA_API_KEY", "exa-test-secret")

    hits = search_mod.web_search("siapa presiden indonesia sekarang", limit=3)
    assert captured["url"] == search_mod.EXA_SEARCH_API
    assert captured["key"] == "exa-test-secret"
    assert captured["body"]["contents"]["highlights"]["numSentences"] == 3
    assert len(hits) == 3

    # Trusted domains (.go.id and antaranews.com) rank ahead of untrusted blog/lookalike
    assert hits[0].url == "https://www.setneg.go.id/presiden-ri"
    assert "(2025-10-20)" in hits[0].title
    assert "Prabowo Subianto" in hits[0].text
    assert hits[1].url == "https://www.antaranews.com/berita/presiden"
    assert "(2026-03-05)" in hits[1].title
    # Duplicate setneg.go.id was skipped for multi-domain diversity; fakereuters.com was NOT promoted as trusted
    assert hits[2].url == "https://blog-acak.example/post-1"
    assert not search_mod._is_trusted_domain("fakereuters.com")
    assert search_mod._is_trusted_domain("setneg.go.id")
    assert search_mod._is_trusted_domain("reuters.com")


def test_multiturn_followup_query_expansion():
    from kuon_agent.agent import AgentConfig, KuonAgent
    from kuon_agent.llm import MockEngine
    from kuon_agent.search import Source

    seen_queries: list[str] = []

    def spy_web(query: str, limit: int = 3) -> list[Source]:
        seen_queries.append(query)
        return [Source(title="Setneg (2024-10-20)", url="https://setneg.go.id", text="Pelantikan 20 Oktober 2024.")]

    agent = KuonAgent(
        MockEngine(),
        AgentConfig(use_rag=False, use_wiki=False, use_web=True, think=False),
        web_fn=spy_web,
    )
    history = [
        ("user", "Siapa presiden Indonesia sekarang?"),
        ("assistant", "Prabowo Subianto."),
        ("user", "Kapan dia dilantik?"),
    ]
    res = agent.answer("Kapan dia dilantik?", history=history)
    assert seen_queries and "presiden Indonesia" in seen_queries[0] and "Kapan dia dilantik?" in seen_queries[0]
    assert res.sources and res.sources[0]["url"] == "https://setneg.go.id"

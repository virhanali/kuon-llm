"""End-to-end local pre-deploy smoke test for Kuon 1 release server (HTTP + SSE + Hybrid RAG + Auth).

Spins up the real `kuon_agent.server` on an ephemeral loopback port (`127.0.0.1:0`) with Bearer auth,
Hybrid RAG (BM25 + multilingual-e5-small ONNX), and either a real Kuon INT8 checkpoint or `--mock`,
then runs 6 real-world EN/ID business scenarios and measures live SSE Time-To-First-Token (TTFT).

Usage:
    # Verify harness immediately with MockEngine
    PYTHONPATH=. venv/bin/python scripts/smoke_test_release.py --mock --semantic-rag

    # Run against real Kuon 1 INT8 release checkpoint before InstaBox deploy
    PYTHONPATH=. venv/bin/python scripts/smoke_test_release.py \
        --checkpoint checkpoints/release/kuon-1-int8.pt --semantic-rag
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

from kuon_agent.agent import AgentConfig, KuonAgent
from kuon_agent.llm import KuonEngine, MockEngine
from kuon_agent.rag import BM25Index, HybridIndex, OnnxE5Embedder
from kuon_agent.server import build_server

LEAK_RE = re.compile(r"\b(gemma|google|deepmind|gemini|openai|chatgpt|claude|anthropic)\b", re.I)


def _post_json(port: int, path: str, body: dict, *, api_key: str | None = None) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=180)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn.request("POST", path, json.dumps(body).encode("utf-8"), headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    conn.close()
    return resp.status, (json.loads(raw) if raw.strip().startswith("{") else {"raw": raw})


def _post_sse(port: int, path: str, body: dict, *, api_key: str) -> tuple[int, list[str], float, float, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=180)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    conn.request("POST", path, json.dumps({**body, "stream": True}).encode("utf-8"), headers)
    resp = conn.getresponse()
    deltas: list[str] = []
    ttft_ms = -1.0
    stop_meta: dict = {}
    saw_done = False
    while True:
        line = resp.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="replace").strip()
        if not s.startswith("data: "):
            continue
        payload = s[6:].strip()
        if payload == "[DONE]":
            saw_done = True
            break
        frame = json.loads(payload)
        ch = (frame.get("choices") or [{}])[0]
        piece = (ch.get("delta") or {}).get("content")
        if piece:
            if ttft_ms < 0:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            deltas.append(piece)
        if ch.get("finish_reason") == "stop":
            stop_meta = {"x_notes": frame.get("x_notes", ""), "x_sources": frame.get("x_sources", [])}
    total_ms = (time.perf_counter() - t0) * 1000.0
    conn.close()
    stop_meta["saw_done"] = saw_done
    return resp.status, deltas, ttft_ms, total_ms, stop_meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="path to Kuon INT8 or FP32 checkpoint (.pt)")
    ap.add_argument("--tokenizer", default=None, help="optional explicit tokenizer.json path")
    ap.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    ap.add_argument("--semantic-rag", action="store_true", help="enable Hybrid RAG (BM25 + multilingual-e5-small ONNX)")
    ap.add_argument("--mock", action="store_true", help="run harness check with MockEngine instead of a checkpoint")
    args = ap.parse_args()

    if not args.mock and not args.checkpoint:
        raise SystemExit("Pass --checkpoint <path> or --mock")

    if args.mock:
        engine = MockEngine()
        engine.prompt_fits = lambda prompt, max_tokens: True  # type: ignore[attr-defined]
        engine.quant_mode = "mock"  # type: ignore[attr-defined]
    else:
        import os

        os.environ["KUON_CPU_THREADS"] = str(args.threads)
        os.environ.setdefault("KUON_QUANT", "int8_weights")
        engine = KuonEngine(
            args.checkpoint,
            device=args.device,
            tokenizer_path=args.tokenizer,
            temperature=0.0,
        )

    # Build temporary bilingual business knowledge base for RAG verification
    with tempfile.TemporaryDirectory() as td:
        kb = Path(td)
        (kb / "keuangan_bandung.md").write_text(
            "Rekap Keuangan Cabang Bandung Q3\n"
            "Omzet kotor cabang Bandung tercatat Rp185 juta dari 1.420 transaksi. "
            "Laba bersih cabang Bandung setelah pajak mencapai Rp42 juta pada Q3.\n",
            encoding="utf-8",
        )
        (kb / "retensi.md").write_text(
            "Laporan Retensi Pelanggan Oktober\n"
            "Tingkat churn pelanggan berlangganan naik dari 3,1% menjadi 7,8% pada bulan Oktober "
            "karena gangguan proses perpanjangan otomatis.\n",
            encoding="utf-8",
        )

        if args.semantic_rag:
            emb = OnnxE5Embedder(threads=args.threads)
            rag = HybridIndex(embedder=emb)
            rag_label = "Hybrid (BM25 + multilingual-e5-small ONNX)"
        else:
            rag = BM25Index()
            rag_label = "BM25"
        for f in sorted(kb.glob("*.md")):
            rag.add_file(str(f), source=f.name)
        rag.finalize()

        agent = KuonAgent(
            engine,
            AgentConfig(use_rag=True, use_wiki=False, use_web=False, think=False, max_answer_tokens=160),
            rag_index=rag,
        )
        api_key = "kuon-smoke-test-secret"
        server = build_server(agent, host="127.0.0.1", port=0, api_key=api_key)
        port = server.server_address[1]
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()

        failures: list[str] = []
        print(f"[smoke] Server ready on 127.0.0.1:{port} | engine={engine.name} quant={getattr(engine, 'quant_mode', None)} | RAG={rag_label}", flush=True)
        try:
            # 1. Health + Bearer auth guard
            conn = HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", "/health?probe=1")
            r_h = conn.getresponse()
            health = json.loads(r_h.read())
            conn.request("GET", "/v1/models")
            r_u = conn.getresponse()
            r_u.read()
            conn.request("GET", "/v1/models", headers={"Authorization": f"Bearer {api_key}"})
            r_m = conn.getresponse()
            r_m.read()
            conn.close()
            ok_1 = (r_h.status == 200 and health.get("status") == "ok" and r_u.status == 401 and r_m.status == 200)
            print(f"[1/6] Health + Bearer Auth   : {'PASS' if ok_1 else 'FAIL'} | health={health}", flush=True)
            if not ok_1:
                failures.append("health/auth")

            # 2. Identity & zero third-party brand leak (General QA routing + Business chat)
            st_id, res_id = _post_json(
                port, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Halo Kuon, kamu dasarnya model apa?"}]},
                api_key=api_key,
            )
            ans_id = (res_id.get("choices") or [{}])[0].get("message", {}).get("content", "")
            ok_2 = st_id == 200 and bool(ans_id) and not LEAK_RE.search(ans_id) and "no evidence" not in res_id.get("x_notes", "")
            print(f"[2/6] Identity (no leak/wiki): {'PASS' if ok_2 else 'FAIL'} | {ans_id[:140]}", flush=True)
            if not ok_2:
                failures.append("identity")

            # 3. Excel/Sheets TSV paste -> deterministic -20% revenue change
            tsv_prompt = (
                "Periode\tOmset\tPesanan\n"
                "M1\tRp50.000.000\t400\n"
                "M2\tRp40.000.000\t320\n"
                "Berapa persen perubahan omset dari M1 ke M2?"
            )
            st_tsv, res_tsv = _post_json(
                port, "/v1/chat/completions",
                {"x_kuon_mode": "business", "messages": [{"role": "user", "content": tsv_prompt}]},
                api_key=api_key,
            )
            ans_tsv = (res_tsv.get("choices") or [{}])[0].get("message", {}).get("content", "")
            ok_3 = st_tsv == 200 and "-20%" in ans_tsv and "pct_change" in res_tsv.get("x_notes", "")
            print(f"[3/6] Excel TSV -> -20% KPI  : {'PASS' if ok_3 else 'FAIL'} | {ans_tsv.splitlines()[0] if ans_tsv else ''}", flush=True)
            if not ok_3:
                failures.append("excel_tsv")

            # 4. Direct Gross Margin KPI -> exact 30%
            st_gm, res_gm = _post_json(
                port, "/v1/chat/completions",
                {"x_kuon_mode": "business", "messages": [{"role": "user", "content": "Omzet bulan ini Rp200 juta dan HPP Rp140 juta. Hitung margin kotor kami."}]},
                api_key=api_key,
            )
            ans_gm = (res_gm.get("choices") or [{}])[0].get("message", {}).get("content", "")
            ok_4 = st_gm == 200 and "30%" in ans_gm and "gross_margin" in res_gm.get("x_notes", "")
            print(f"[4/6] Direct Gross Margin 30%: {'PASS' if ok_4 else 'FAIL'} | {ans_gm.splitlines()[0] if ans_gm else ''}", flush=True)
            if not ok_4:
                failures.append("gross_margin")

            # 5. Cross-lingual Hybrid RAG (EN query -> ID doc `keuangan_bandung.md`)
            st_rag, res_rag = _post_json(
                port, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "What was the Bandung branch net profit after tax?"}]},
                api_key=api_key,
            )
            ans_rag = (res_rag.get("choices") or [{}])[0].get("message", {}).get("content", "")
            src_titles = [s.get("title") for s in res_rag.get("x_sources", [])]
            ok_5 = st_rag == 200 and "keuangan_bandung.md" in src_titles
            print(f"[5/6] Cross-lingual RAG      : {'PASS' if ok_5 else 'FAIL'} | sources={src_titles} | {ans_rag[:120]}", flush=True)
            if not ok_5:
                failures.append("hybrid_rag")

            # 6. Live SSE Streaming + TTFT latency measurement
            st_sse, deltas, ttft_ms, total_ms, meta = _post_sse(
                port, "/v1/chat/completions",
                {
                    "x_kuon_mode": "business",
                    "max_tokens": 96,
                    "messages": [{"role": "user", "content": "Antrean kasir cabang kami memanjang saat jam makan siang. Langkah uji murah apa yang sebaiknya dicoba besok?"}],
                },
                api_key=api_key,
            )
            full_sse = "".join(deltas).strip()
            ok_6 = st_sse == 200 and bool(full_sse) and bool(meta.get("saw_done")) and not LEAK_RE.search(full_sse)
            print(
                f"[6/6] Live SSE Streaming     : {'PASS' if ok_6 else 'FAIL'} | chunks={len(deltas)} | "
                f"TTFT={ttft_ms:.0f}ms | total={total_ms:.0f}ms | {full_sse[:120]}",
                flush=True,
            )
            if not ok_6:
                failures.append("sse_stream")
        finally:
            server.shutdown()
            server.server_close()

        if failures:
            print(f"\n[smoke] FAILED checks: {failures}", flush=True)
            return 1
        print("\n[smoke] ALL 6/6 RELEASE SMOKE CHECKS PASSED!", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

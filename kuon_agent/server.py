"""OpenAI-compatible HTTP API + chat UI for Kuon Agent (stdlib only).

Endpoints:
    GET  /                       single-page chat UI (the Kuon Studio front page)
    GET  /v1/models              OpenAI-style model list
    POST /v1/chat/completions    OpenAI chat completions (stream=false|true)
    GET  /health                 liveness probe

Compatible clients can point at it with a base URL of http://host:8000/v1
and any API key (the server ignores Authorization). Extra fields
`x_sources` and `x_notes` are added to responses for the UI; standard
clients simply ignore them.

Run:
    python -m kuon_agent.server --engine ollama --model qwen2.5-coder:7b --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kuon_agent.agent import AgentConfig, KuonAgent
from kuon_agent.llm import KuonEngine, LlamaCppEngine, MockEngine, OllamaEngine
from kuon_agent.rag import BM25Index

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kuon Studio</title>
<style>
  :root { --bg:#0b0e14; --panel:#121722; --line:#1f2733; --text:#e6edf3; --dim:#8b98a9; --accent:#7c9cff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.6 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; }
  header h1 { margin:0; font-size:18px; letter-spacing:.4px; }
  header span { color:var(--dim); font-size:13px; }
  main { max-width:820px; margin:0 auto; padding:22px 16px 140px; }
  .msg { margin:14px 0; display:flex; gap:10px; }
  .msg .who { color:var(--dim); font-size:12px; min-width:64px; padding-top:4px; text-transform:uppercase; letter-spacing:.8px; }
  .bubble { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; white-space:pre-wrap; flex:1; }
  .msg.user .bubble { background:#101827; }
  .sources { margin-top:10px; font-size:13px; color:var(--dim); }
  .sources a { color:var(--accent); text-decoration:none; }
  .sources div { margin-top:4px; }
  footer { position:fixed; bottom:0; left:0; right:0; background:linear-gradient(transparent, var(--bg) 30%); padding:20px 16px 24px; }
  .composer { max-width:820px; margin:0 auto; display:flex; gap:10px; }
  input[type=text] { flex:1; background:var(--panel); border:1px solid var(--line); color:var(--text); border-radius:12px; padding:14px 16px; font-size:16px; outline:none; }
  input[type=text]:focus { border-color:#33415c; }
  button { background:var(--accent); color:#0b0e14; border:0; border-radius:12px; padding:0 22px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .hint { max-width:820px; margin:8px auto 0; color:var(--dim); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Kuon Studio</h1>
  <span>from-scratch agent &mdash; search &rarr; think &rarr; answer, with sources</span>
</header>
<main id="chat">
  <div class="msg assistant">
    <div class="who">Kuon</div>
    <div class="bubble">Halo! Tanyakan apa saja. Aku mencari bukti dulu (Wikipedia / index lokal), lalu menjawab dari bukti itu dengan sitasi.</div>
  </div>
</main>
<footer>
  <div class="composer">
    <input id="q" type="text" placeholder="who is mr beast?" autocomplete="off">
    <button id="send">Kirim</button>
  </div>
  <div class="hint">Jawaban hanya dari sumber. Tanpa bukti, Kuon akan bilang tidak tahu.</div>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('q');
const send = document.getElementById('send');

function addMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  wrap.innerHTML = '<div class="who">' + (role === 'user' ? 'Kamu' : 'Kuon') + '</div>' +
                   '<div class="bubble"></div>';
  wrap.querySelector('.bubble').textContent = text;
  chat.appendChild(wrap);
  window.scrollTo(0, document.body.scrollHeight);
  return wrap.querySelector('.bubble');
}

function addSources(bubble, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement('div');
  box.className = 'sources';
  sources.forEach((s, i) => {
    const line = document.createElement('div');
    const label = document.createElement('span');
    label.textContent = '[' + (i + 1) + '] ' + s.title;
    line.appendChild(label);
    if (s.url) {
      const a = document.createElement('a');
      a.href = s.url; a.textContent = ' - ' + s.url; a.target = '_blank';
      line.appendChild(a);
    }
    box.appendChild(line);
  });
  bubble.appendChild(box);
}

async function ask() {
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  send.disabled = true;
  addMessage('user', question);
  const bubble = addMessage('assistant', '…');
  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'kuon-agent', messages: [{role: 'user', content: question}]})
    });
    const data = await response.json();
    bubble.textContent = data.choices[0].message.content;
    addSources(bubble, data.x_sources || []);
  } catch (error) {
    bubble.textContent = 'Error: ' + error;
  } finally {
    send.disabled = false;
    input.focus();
  }
}

send.addEventListener('click', ask);
input.addEventListener('keydown', (event) => { if (event.key === 'Enter') ask(); });
input.focus();
</script>
</body>
</html>
"""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _content_of(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):  # OpenAI multimodal shape: keep text parts
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def build_handler(agent, model_name: str = "kuon-agent"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # ------------------------------------------------------------ utils

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------------ verbs

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": model_name, "object": "model", "owned_by": "kuon"}],
                })
            elif self.path == "/health":
                self._json(200, {"status": "ok", "model": model_name})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                messages = payload.get("messages", [])
                question = next((_content_of(m) for m in reversed(messages)
                                 if m.get("role") == "user"), "")
                if not question:
                    self._json(400, {"error": {"message": "no user message"}})
                    return

                result = agent.answer(question)
                if payload.get("stream"):
                    self._stream(result, question)
                else:
                    self._completion(result, question)
            except Exception as error:  # never leak tracebacks to clients
                self._json(500, {"error": {"message": str(error)}})

        # --------------------------------------------------------- responses

        def _completion(self, result, question: str):
            prompt_tokens = _estimate_tokens(question)
            completion_tokens = _estimate_tokens(result.answer)
            self._json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answer},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "x_sources": result.sources,
                "x_notes": result.notes,
            })

        def _stream(self, result, question: str):
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

            def chunk(delta: dict, finish=None):
                frame = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                self.wfile.flush()

            chunk({"role": "assistant"})
            chunk({"content": result.answer})
            chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def build_server(agent, host: str = "127.0.0.1", port: int = 8000,
                 model_name: str = "kuon-agent") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), build_handler(agent, model_name))


def agent_from_env(args) -> KuonAgent:
    engine_name = args.engine or os.environ.get("KUON_ENGINE", "mock")
    if engine_name == "mock":
        engine = MockEngine()
    elif engine_name == "kuon":
        checkpoint = args.checkpoint or os.environ.get("KUON_CHECKPOINT", "")
        if not checkpoint:
            raise SystemExit("--engine kuon requires --checkpoint or KUON_CHECKPOINT")
        engine = KuonEngine(checkpoint, device=args.device)
    elif engine_name == "ollama":
        engine = OllamaEngine(
            model=args.model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            host=args.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    elif engine_name == "llamacpp":
        model = args.model or os.environ.get("KUON_GGUF", "")
        if not model:
            raise SystemExit("--engine llamacpp requires --model or KUON_GGUF")
        engine = LlamaCppEngine(model, binary=args.binary)
    else:
        raise SystemExit(f"unknown engine {engine_name}")

    rag_path = args.rag or os.environ.get("KUON_RAG")
    rag_index = None
    if rag_path:
        from pathlib import Path

        rag_index = BM25Index()
        target = Path(rag_path)
        if target.is_dir():
            for file in sorted(target.glob("**/*")):
                if file.suffix.lower() in {".txt", ".md"}:
                    rag_index.add_file(str(file), source=str(file))
        else:
            rag_index.add_file(str(target), source=target.name)
        rag_index.finalize()

    return KuonAgent(engine, AgentConfig(
        use_rag=bool(rag_path),
        use_wiki=not args.no_wiki,
        use_web=args.web,
        fallback_web=args.web,
        think=not args.no_think,
    ), rag_index=rag_index)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kuon Agent OpenAI-compatible server")
    ap.add_argument("--engine", choices=["mock", "kuon", "ollama", "llamacpp"], default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None, help="Ollama model or GGUF path")
    ap.add_argument("--host", default=None, help="Ollama host (http://localhost:11434)")
    ap.add_argument("--binary", default="llama-cli")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rag", default=None, help="file or directory for BM25 RAG")
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--web", action="store_true", help="enable DuckDuckGo search")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    agent = agent_from_env(args)
    server = build_server(agent, args.bind, args.port)
    print(f"[kuon] serving on http://{args.bind}:{args.port}  (engine: {agent.engine.name})")
    print("[kuon] OpenAI-compatible base URL: http://host:port/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

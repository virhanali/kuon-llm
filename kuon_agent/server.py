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
import hmac
import inspect
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kuon_agent.agent import AgentConfig, ContextLimitError, KuonAgent
from kuon_agent.llm import KuonEngine, LlamaCppEngine, MockEngine, OllamaEngine
from kuon_agent.rag import BM25Index, HybridIndex, OnnxE5Embedder

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
  header { padding:18px 22px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:18px; letter-spacing:.4px; }
  header span { color:var(--dim); font-size:13px; }
  .controls { margin-left:auto; display:flex; gap:8px; align-items:center; }
  .controls select, .controls button { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:8px; padding:6px 10px; font-size:13px; cursor:pointer; }
  main { max-width:820px; margin:0 auto; padding:22px 16px 160px; }
  .msg { margin:14px 0; display:flex; gap:10px; }
  .msg .who { color:var(--dim); font-size:12px; min-width:64px; padding-top:4px; text-transform:uppercase; letter-spacing:.8px; }
  .bubble { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; white-space:pre-wrap; flex:1; }
  .msg.user .bubble { background:#101827; }
  .notes { margin-top:10px; padding:8px 10px; background:#0d131d; border:1px solid var(--line); border-radius:8px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:#9db4d8; }
  .sources { margin-top:8px; font-size:13px; color:var(--dim); }
  .sources a { color:var(--accent); text-decoration:none; }
  .sources div { margin-top:4px; }
  footer { position:fixed; bottom:0; left:0; right:0; background:linear-gradient(transparent, var(--bg) 30%); padding:20px 16px 24px; }
  .composer { max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; background:var(--panel); border:1px solid var(--line); color:var(--text); border-radius:12px; padding:12px 14px; font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif; outline:none; resize:vertical; min-height:52px; max-height:180px; }
  textarea:focus { border-color:#33415c; }
  .composer button { background:var(--accent); color:#0b0e14; border:0; border-radius:12px; padding:14px 22px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .hint { max-width:820px; margin:8px auto 0; color:var(--dim); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Kuon Studio</h1>
  <span>from-scratch business &amp; evidence agent &mdash; verify &rarr; compute &rarr; answer</span>
  <div class="controls">
    <select id="mode" aria-label="Chat mode">
      <option value="business" selected>Business chat</option>
      <option value="general">General QA</option>
    </select>
    <button id="reset" type="button">Reset</button>
  </div>
</header>
<main id="chat">
  <div class="msg assistant">
    <div class="who">Kuon</div>
    <div class="bubble">Halo! Tempel tabel penjualan atau data bisnis Anda. Kuon memeriksa angka, menghitung dengan rumus pasti, dan menampilkan sumber buktinya.</div>
  </div>
</main>
<footer>
  <div class="composer">
    <textarea id="q" rows="2" placeholder="Tempel tabel omzet atau tanyakan KPI bisnis (Shift+Enter untuk baris baru)..."></textarea>
    <button id="send">Kirim</button>
  </div>
  <div class="hint">Jawaban hanya dari bukti dan alat hitung. Tanpa data yang cukup, Kuon akan meminta klarifikasi.</div>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('q');
const send = document.getElementById('send');
const modeSel = document.getElementById('mode');
const resetBtn = document.getElementById('reset');
const history = [];

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

function addProvenance(bubble, notes, sources) {
  if (notes) {
    const n = document.createElement('div');
    n.className = 'notes';
    n.textContent = notes;
    bubble.appendChild(n);
  }
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
  if (!question || send.disabled) return;
  input.value = '';
  send.disabled = true;
  addMessage('user', question);
  history.push({role: 'user', content: question});
  const bubble = addMessage('assistant', '…');
  try {
    const req = {model: 'kuon-agent', messages: history.slice(), stream: true};
    if (modeSel.value === 'business') req.x_kuon_mode = 'business';
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      history.pop();
      bubble.textContent = data.error?.message || 'Could not answer. Please try again.';
      return;
    }
    const ctype = response.headers.get('Content-Type') || '';
    if (ctype.includes('text/event-stream') && response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = '', content = '', notes = '', sources = [];
      bubble.textContent = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buf += decoder.decode(value, {stream: true});
        let nl;
        while ((nl = buf.indexOf('\n\n')) !== -1) {
          const block = buf.slice(0, nl);
          buf = buf.slice(nl + 2);
          for (const line of block.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            const raw = line.slice(6).trim();
            if (!raw || raw === '[DONE]') continue;
            const frame = JSON.parse(raw);
            const delta = frame.choices?.[0]?.delta?.content;
            if (delta) {
              content += delta;
              bubble.textContent = content;
              window.scrollTo(0, document.body.scrollHeight);
            }
            if (frame.x_notes) notes = frame.x_notes;
            if (frame.x_sources) sources = frame.x_sources;
          }
        }
      }
      history.push({role: 'assistant', content});
      addProvenance(bubble, notes, sources);
    } else {
      const data = await response.json();
      const content = data.choices[0].message.content;
      history.push({role: 'assistant', content});
      bubble.textContent = content;
      addProvenance(bubble, data.x_notes || '', data.x_sources || []);
    }
  } catch (error) {
    history.pop();
    bubble.textContent = 'Error: ' + error;
  } finally {
    send.disabled = false;
    input.focus();
  }
}

resetBtn.addEventListener('click', () => {
  history.length = 0;
  chat.innerHTML = '';
  addMessage('assistant', 'Percakapan direset. Silakan kirim tabel atau pertanyaan bisnis baru.');
  input.focus();
});
send.addEventListener('click', ask);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});
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


def _call_compat(fn, *args, **kwargs):
    """Forward optional kwargs only if `fn` accepts them (keeps test stubs compatible)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)
    params = sig.parameters
    accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    filtered = kwargs if accepts_var else {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def build_handler(agent, model_name: str = "kuon-agent",
                  agents: dict | None = None,
                  *, api_key: str | None = None):
    """agents maps model ids -> KuonAgent; model_name is the default id."""
    registry = dict(agents) if agents else {model_name: agent}
    inference_lock = threading.Lock()
    state_lock = threading.Lock()
    runtime_state = {"active": 0, "queued": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # ------------------------------------------------------------ utils

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _authorized(self) -> bool:
            if not api_key:
                return True
            hdr = self.headers.get("Authorization", "").strip()
            if not hdr.lower().startswith("bearer "):
                return False
            token = hdr[7:].strip()
            return hmac.compare_digest(token, api_key)

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
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/v1/models":
                if not self._authorized():
                    self._json(401, {"error": {"message": "unauthorized", "code": "invalid_api_key"}})
                    return
                self._json(200, {
                    "object": "list",
                    "data": [{"id": name, "object": "model", "owned_by": "kuon"}
                             for name in registry],
                })
            elif path == "/health":
                with state_lock:
                    active = runtime_state["active"]
                    queued = runtime_state["queued"]
                eng = getattr(agent, "engine", None)
                self._json(200, {
                    "status": "ok",
                    "model": model_name,
                    "engine": getattr(eng, "name", "unknown"),
                    "quant": getattr(eng, "quant_mode", None),
                    "busy": active > 0,
                    "queue_depth": queued,
                })
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            if not self._authorized():
                self._json(401, {"error": {"message": "unauthorized", "code": "invalid_api_key"}})
                return
            sse_started = False
            try:
                raw_len = self.headers.get("Content-Length", "0")
                try:
                    length = int(raw_len)
                except ValueError:
                    self._json(400, {"error": {"message": "invalid Content-Length header"}})
                    return
                if length < 0:
                    self._json(400, {"error": {"message": "invalid Content-Length header"}})
                    return
                if length > 1_048_576:
                    self._json(413, {"error": {"message": "request body exceeds 1 MB limit"}})
                    return
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._json(400, {"error": {"message": "invalid JSON body"}})
                    return
                if not isinstance(payload, dict):
                    self._json(400, {"error": {"message": "request body must be a JSON object"}})
                    return
                messages = payload.get("messages", [])
                if not isinstance(messages, list):
                    self._json(400, {"error": {"message": "messages must be a list"}})
                    return
                user_index = next((i for i in range(len(messages) - 1, -1, -1)
                                   if isinstance(messages[i], dict) and messages[i].get("role") == "user"), -1)
                question = _content_of(messages[user_index]).strip() if user_index >= 0 else ""
                if not question:
                    self._json(400, {"error": {"message": "no user message"}})
                    return

                requested = payload.get("model") or model_name
                chosen = registry.get(requested, agent)
                wants_stream = bool(payload.get("stream"))
                try:
                    req_temp = payload.get("temperature")
                    temperature = max(0.0, min(1.5, float(req_temp))) if req_temp is not None else None
                    req_max = payload.get("max_tokens")
                    max_tokens = max(1, min(512, int(req_max))) if req_max is not None else None
                except (TypeError, ValueError):
                    self._json(400, {"error": {"message": "invalid temperature or max_tokens"}})
                    return

                completion_id = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                emitted_any = False

                def start_sse():
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self._cors()
                    self.end_headers()
                    send_chunk({"role": "assistant"})

                def send_chunk(delta: dict, finish=None, extra: dict | None = None):
                    frame = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": requested or model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    if extra:
                        frame.update(extra)
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                    self.wfile.flush()

                def on_token(piece: str):
                    nonlocal sse_started, emitted_any
                    if not piece:
                        return
                    if not sse_started:
                        start_sse()
                        sse_started = True
                    send_chunk({"content": piece})
                    emitted_any = True

                call_kwargs = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "on_token": (on_token if wants_stream else None),
                }

                acquired = False
                with state_lock:
                    runtime_state["queued"] += 1
                try:
                    with inference_lock:
                        acquired = True
                        with state_lock:
                            runtime_state["queued"] -= 1
                            runtime_state["active"] += 1
                        try:
                            if payload.get("x_kuon_mode") == "business":
                                history = [(m["role"], _content_of(m).strip())
                                           for m in messages[:user_index + 1]
                                           if m.get("role") in ("user", "assistant")]
                                try:
                                    result = _call_compat(chosen.advise, history, **call_kwargs)
                                except ContextLimitError:
                                    if not sse_started:
                                        self._json(400, {"error": {
                                            "code": "context_length_exceeded",
                                            "message": "The latest message exceeds the token budget. Shorten it and try again.",
                                        }})
                                    return
                                except NotImplementedError:
                                    if not sse_started:
                                        self._json(501, {"error": {
                                            "message": "Business chat is unavailable on the selected engine.",
                                        }})
                                    return
                            else:
                                qa_history = [(m["role"], _content_of(m).strip())
                                              for m in messages[:user_index + 1]
                                              if m.get("role") in ("user", "assistant")]
                                result = _call_compat(chosen.answer, question, history=qa_history, **call_kwargs)
                        finally:
                            with state_lock:
                                runtime_state["active"] -= 1
                finally:
                    if not acquired:
                        with state_lock:
                            runtime_state["queued"] -= 1

                if wants_stream:
                    if not sse_started:
                        start_sse()
                        sse_started = True
                    if not emitted_any and result.answer:
                        send_chunk({"content": result.answer})
                    send_chunk({}, "stop", {"x_sources": result.sources, "x_notes": result.notes})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self._completion(result, question, requested)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:  # never leak tracebacks to clients
                if not sse_started:
                    self._json(500, {"error": {"message": str(error)}})

        # --------------------------------------------------------- responses

        def _completion(self, result, question: str, model_id: str = None):
            prompt_tokens = _estimate_tokens(question)
            completion_tokens = _estimate_tokens(result.answer)
            self._json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id or model_name,
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

    return Handler


def build_server(agent, host: str = "127.0.0.1", port: int = 8000,
                 model_name: str = "kuon-agent",
                 agents: dict | None = None,
                 *, api_key: str | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), build_handler(agent, model_name, agents, api_key=api_key))


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

        use_semantic = bool(getattr(args, "semantic_rag", False)) or (
            os.environ.get("KUON_SEMANTIC_RAG", "").strip().lower() in ("1", "true", "yes", "on")
        )
        if use_semantic:
            try:
                threads_env = os.environ.get("KUON_CPU_THREADS", "4")
                threads = int(threads_env) if threads_env.isdigit() else 4
                embedder = OnnxE5Embedder(threads=threads)
                rag_index = HybridIndex(embedder=embedder)
                print("[kuon] Hybrid RAG enabled (BM25 + multilingual-e5-small ONNX)", flush=True)
            except Exception as err:
                print(f"[kuon] Hybrid RAG unavailable ({err}); falling back to BM25", flush=True)
                rag_index = BM25Index()
        else:
            rag_index = BM25Index()
        target = Path(rag_path)
        if target.is_dir():
            for file in sorted(target.glob("**/*")):
                if file.suffix.lower() in {".txt", ".md", ".csv", ".tsv"}:
                    rag_index.add_file(str(file), source=str(file))
        else:
            rag_index.add_file(str(target), source=target.name)
        rag_index.finalize()

    has_exa = bool(os.environ.get("EXA_API_KEY", "").strip())
    use_web = bool(args.web or has_exa)
    if has_exa:
        print("[kuon] Live web search enabled via Exa AI (General QA mode only)", flush=True)
    return KuonAgent(engine, AgentConfig(
        use_rag=bool(rag_path),
        use_wiki=not args.no_wiki,
        use_web=use_web,
        fallback_web=use_web,
        think=False,
    ), rag_index=rag_index)


def _load_dotenv() -> None:
    """Load simple KEY=VALUE lines from local .env without overriding existing env vars."""
    from pathlib import Path

    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        resolved = env_path.resolve()
        if resolved in seen or not env_path.is_file():
            continue
        seen.add(resolved)
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Kuon Agent OpenAI-compatible server")
    ap.add_argument("--engine", choices=["mock", "kuon", "ollama", "llamacpp"], default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None, help="Ollama model or GGUF path")
    ap.add_argument("--host", default=None, help="Ollama host (http://localhost:11434)")
    ap.add_argument("--binary", default="llama-cli")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rag", default=None, help="file or directory for RAG (.txt, .md, .csv, .tsv)")
    ap.add_argument("--semantic-rag", action="store_true",
                    help="combine BM25 with MIT multilingual-e5-small ONNX semantic search (or set KUON_SEMANTIC_RAG=1)")
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--web", action="store_true", help="enable DuckDuckGo search")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--ollama-model", default=None,
                    help="register a second engine served as model id 'qwen' (e.g. qwen2.5:3b)")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    agent = agent_from_env(args)
    agents = {"kuon-agent": agent}
    extra_model = args.ollama_model or os.environ.get("KUON_OLLAMA_MODEL")
    if extra_model:
        agents["qwen"] = KuonAgent(OllamaEngine(model=extra_model),
                                   agent.cfg, rag_index=agent.rag)
    api_key = os.environ.get("KUON_API_KEY") or None
    server = build_server(agent, args.bind, args.port, agents=agents, api_key=api_key)
    print(f"[kuon] serving on http://{args.bind}:{args.port}  (engine: {agent.engine.name})")
    print(f"[kuon] models: {', '.join(agents)} | auth: {'bearer' if api_key else 'open'}")
    print("[kuon] OpenAI-compatible base URL: http://host:port/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

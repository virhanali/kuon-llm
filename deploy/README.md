# Kuon Studio deployment

Two supported paths:

- **VPS (2 vCPU / 8 GB is enough)** — see [vps.md](vps.md); everything runs on
  CPU with Ollama + Qwen 2.5 3B. Includes systemd unit and Caddyfile.
- **Cloudflare Tunnel** — free, no VPS: run the server anywhere and expose it
  with `cloudflared tunnel --url http://localhost:8000`.

The server is dependency-free (Python stdlib). Any small box works.

## Local

```bash
python -m kuon_agent.server --engine ollama --model qwen2.5-coder:7b --port 8000
# open http://localhost:8000
```

OpenAI-compatible base URL: `http://localhost:8000/v1` (any API key).

## Docker

```bash
docker build -t kuon-studio -f deploy/Dockerfile .
docker run -p 8000:8000 -e KUON_ENGINE=ollama kuon-studio
```

## Free hosting options (this app needs a persistent Python process)

Static hosts (Vercel/Netlify/Cloudflare Pages) are NOT enough: the agent is a
server. Free options that run it:

| Option | Free? | Always on? | Custom domain? | Notes |
|---|---|---|---|---|
| **Hugging Face Spaces** (Docker SDK) | yes — 2 vCPU / 16 GB CPU | yes | paid (PRO) | Best free portfolio host; URL `*.hf.space`; `app_port: 8000` |
| **Cloudflare Tunnel** | yes | only while your machine runs | yes (if domain on Cloudflare) | Zero cost, instant; `cloudflared tunnel --url http://localhost:8000` |
| **Render** (free web service) | yes (750 h/month) | sleeps after inactivity | yes | Cold start ~30–60 s; Python native |
| **Oracle Cloud Always Free** (ARM VM) | yes — 4 OCPU / 24 GB | yes | yes (via DNS) | Real 24/7 VPS; signup/capacity can be friction |
| **Google Cloud Run** | generous free quota | scales to zero | yes | Needs a billing account (card) even on free quota |
| PythonAnywhere free | yes | yes | no | Outbound internet limited to allow-listed hosts (DDG may fail) |

Recommended:
- **Demo now:** Cloudflare Tunnel from your laptop (free, one command).
- **Portfolio permanent:** Hugging Face Spaces (free, stable URL, no card) —
  add this metadata to a Space repo:

```yaml
---
title: Kuon Studio
sdk: docker
app_port: 8000
---
```
  then push our `deploy/Dockerfile` build as the Space (or the repo with
  `Dockerfile` at the root and `KUON_ENGINE=mock` by default).
- **kuonstudio.com**: a domain costs ~$10/year (domains are the one thing that
  is never free); point it at Cloudflare Tunnel, Render, or an Oracle VM.

## kuonstudio.com (fastest path: Cloudflare Tunnel)

1. Run the server on any machine that can reach your model (e.g. laptop or VPS):
   `python -m kuon_agent.server --bind 0.0.0.0 --port 8000`
2. Install cloudflared and expose it:
   `cloudflared tunnel --url http://localhost:8000`
3. In Cloudflare, point `kuonstudio.com` at the tunnel (named tunnel for a
   stable URL), or add a proxied DNS record to the tunnel hostname.

For a permanent portfolio site, a $5 VPS with Caddy in front of the server is
the simplest stable setup:

```
kuonstudio.com {
    reverse_proxy 127.0.0.1:8000
}
```

## Notes

- Engine is chosen with `KUON_ENGINE` (`mock`, `kuon`, `ollama`, `llamacpp`).
- `KUON_CHECKPOINT`/`KUON_GGUF`/`OLLAMA_MODEL` select the model weights.
- `KUON_RAG` points at a file or directory to index with BM25 (local RAG).
- The agent adds `x_sources` and `x_notes` to OpenAI responses; standard
  clients ignore those fields.

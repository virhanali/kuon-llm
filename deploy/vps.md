# Kuon Studio on a 2 vCPU / 8 GB VPS

Everything runs on CPU. Recommended engine: Ollama + Qwen 2.5 3B (Q4).

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git caddy
```

## 2. Ollama + model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b          # ~2 GB, comfortable on 8 GB
# optional, if you want more quality (tight on RAM):
# ollama pull qwen2.5-coder:7b
```

## 3. Kuon repo + server

```bash
git clone https://github.com/virhanali/kuon-llm.git
cd kuon-llm
python3 -m venv venv
venv/bin/pip install -e .
```

Quick test:

```bash
venv/bin/python -m kuon_agent.server --engine ollama --model qwen2.5:3b --port 8000
curl -s localhost:8000/health
```

## 4. systemd service

```bash
sudo cp deploy/kuon-studio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kuon-studio
```

## 5. Caddy (kuonstudio.com + TLS otomatis)

Point the domain's A record at the VPS IP first, then:

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy obtains the certificate automatically. Open https://kuonstudio.com.

## Memory tips (8 GB box)

- Keep `OLLAMA_NUM_PARALLEL=1` and a small context (`--ctx 4096`).
- Prefer 3B models; 7B Q4 works but leaves little headroom.
- The agent itself is tiny (stdlib, <100 MB).

## Swap (recommended)

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

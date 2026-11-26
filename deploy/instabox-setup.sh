#!/usr/bin/env bash
# Lean InstaBox / VPS CPU deployment script for Kuon 1 (295.8M, 4 vCPU / 8 GiB RAM / 20 GiB Disk).
#
# Installs CPU-only PyTorch (~250 MB, NOT the 4 GB CUDA wheel), pulls the portable INT8
# Kuon 1 checkpoint (~307 MB) from a private Kaggle dataset via KAGGLE_API_TOKEN, and starts
# the OpenAI-compatible Kuon Studio server on 0.0.0.0:8000 with 4 CPU threads.
#
# Usage (on InstaBox Linux medium machine with Auto-pause enabled):
#   Set in InstaBox Secrets: KAGGLE_API_TOKEN, KUON_API_KEY, KUON_SEMANTIC_RAG=1, EXA_API_KEY
#   export KUON_KAGGLE_DATASET="virhanali/kuon2-300m-release"
#   bash deploy/instabox-setup.sh
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/kuon-llm}"
CKPT_DIR="${CKPT_DIR:-$APP_DIR/checkpoints/release}"
CKPT_FILE="${CKPT_FILE:-kuon-2-int8.pt}"
PORT="${PORT:-8000}"
THREADS="${KUON_CPU_THREADS:-4}"

echo "[instabox] Disk headroom before install:"
df -h "$HOME"

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/virhanali/kuon-llm.git "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only
fi
cd "$APP_DIR"

if [[ ! -d "$APP_DIR/venv" ]]; then
  python3 -m venv "$APP_DIR/venv"
fi

echo "[instabox] Installing CPU-only PyTorch (~250 MB), tokenizers, onnxruntime, and Kuon package..."
"$APP_DIR/venv/bin/pip" install --no-cache-dir --upgrade pip
"$APP_DIR/venv/bin/pip" install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
"$APP_DIR/venv/bin/pip" install --no-cache-dir tokenizers onnxruntime
"$APP_DIR/venv/bin/pip" install --no-cache-dir --no-deps -e .

mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT_DIR/$CKPT_FILE" ]]; then
  if [[ -z "${KUON_KAGGLE_DATASET:-}" || -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "[instabox] Checkpoint $CKPT_DIR/$CKPT_FILE missing."
    echo "[instabox] Set KUON_KAGGLE_DATASET and KAGGLE_API_TOKEN (in InstaBox Secrets), or copy checkpoint manually."
    exit 1
  fi
  "$APP_DIR/venv/bin/pip" install --no-cache-dir kaggle
  mkdir -p "$HOME/.kaggle"
  printf '%s\n' "$KAGGLE_API_TOKEN" > "$HOME/.kaggle/access_token"
  chmod 600 "$HOME/.kaggle/access_token"
  "$APP_DIR/venv/bin/kaggle" datasets download -d "$KUON_KAGGLE_DATASET" -f "$CKPT_FILE" -p "$CKPT_DIR" --unzip
  "$APP_DIR/venv/bin/kaggle" datasets download -d "$KUON_KAGGLE_DATASET" -f tokenizer.json -p "$CKPT_DIR" --unzip
fi

echo "[instabox] Disk usage after install:"
df -h "$HOME"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export KUON_CPU_THREADS="$THREADS"
export KUON_QUANT="${KUON_QUANT:-int8_weights}"
export KUON_ENGINE="kuon"
export KUON_CHECKPOINT="$CKPT_DIR/$CKPT_FILE"

AUTH_MSG="open (set KUON_API_KEY in Secrets to enable Bearer auth)"
if [[ -n "${KUON_API_KEY:-}" ]]; then
  AUTH_MSG="Bearer API key enabled"
fi
WEB_MSG="wikipedia-only (set EXA_API_KEY in Secrets to enable Exa live web search)"
if [[ -n "${EXA_API_KEY:-}" ]]; then
  WEB_MSG="Exa AI live search enabled (General QA)"
fi
RAG_ARGS=()
RAG_MSG="disabled (set KUON_RAG=/path/to/docs to index documents)"
if [[ -n "${KUON_RAG:-}" ]]; then
  RAG_ARGS+=(--rag "$KUON_RAG")
  if [[ "${KUON_SEMANTIC_RAG:-0}" == "1" || "${KUON_SEMANTIC_RAG:-}" == "true" ]]; then
    RAG_ARGS+=(--semantic-rag)
    RAG_MSG="Hybrid RAG (BM25 + multilingual-e5-small ONNX) on $KUON_RAG"
  else
    RAG_MSG="BM25 on $KUON_RAG"
  fi
fi
echo "[instabox] Starting Kuon 1 Studio on 0.0.0.0:$PORT (threads=$THREADS, quant=$KUON_QUANT, auth=$AUTH_MSG, web=$WEB_MSG, rag=$RAG_MSG)"
exec "$APP_DIR/venv/bin/python" -m kuon_agent.server \
  --engine kuon \
  --checkpoint "$KUON_CHECKPOINT" \
  --device cpu \
  --bind 0.0.0.0 \
  --port "$PORT" \
  "${RAG_ARGS[@]}"

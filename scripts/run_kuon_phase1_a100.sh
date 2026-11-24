#!/usr/bin/env bash
# Phase 1 Continual Pretraining of Kuon 1 (Gated-Loop 24L) from 307M -> 1.3B tokens on Colab Pro A100.
#
# Designed to be 100% resumable across sessions:
#   - First run uses --init-from (loads Stage 2 kuon-1-loop-best.pt weights with fresh LR schedule)
#   - Subsequent runs auto-detect kuon-1-loop-last.pt in OUT_DIR/checkpoints and pass --resume!
#
# Example on Colab Pro A100 (40GB/80GB, native BF16 + torch.compile):
#   BASE_CKPT=/content/drive/MyDrive/kuon/kuon-1-loop-best.pt \
#   DATA_DIR=/content/kuon1-data-1b-part3 \
#   OUT_DIR=/content/drive/MyDrive/kuon/phase1-1b \
#   bash scripts/run_kuon_phase1_a100.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to clean Stage 2 kuon-1-loop-best.pt}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to packed 1B bilingual corpus dir}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/kuon/phase1-1b}"

# 60/40 bilingual requires grad_accum divisible by 5.
# Target 61,440 tokens/step identically across GPUs -> 16,280 steps ~= 1.00B fresh tokens!
# Auto-scale microbatch for 16GB T4 (BS=6, ACCUM=10) vs 24GB+ L4/A100/G4 (BS=12, ACCUM=5).
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)"
if [[ "${VRAM_MB:-0}" -gt 0 && "${VRAM_MB:-0}" -lt 20000 ]]; then
  DEFAULT_BS=6
  DEFAULT_ACCUM=10
  DEFAULT_EVAL=250
else
  DEFAULT_BS=12
  DEFAULT_ACCUM=5
  DEFAULT_EVAL=500
fi
STEPS="${STEPS:-16280}"
BS="${BS:-$DEFAULT_BS}"
ACCUM="${ACCUM:-$DEFAULT_ACCUM}"
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-1.5e-5}"
WARMUP="${WARMUP:-300}"
EVAL_INT="${EVAL_INT:-$DEFAULT_EVAL}"

CKPT_DIR="$OUT_DIR/checkpoints"
LOG_FILE="$OUT_DIR/logs/kuon-1-loop-1b.jsonl"
mkdir -p "$CKPT_DIR" "$(dirname "$LOG_FILE")"

RESUME_CKPT="$CKPT_DIR/kuon-1-loop-last.pt"
INIT_ARGS=()
if [ -f "$RESUME_CKPT" ]; then
  echo "[phase1-a100] Resuming existing Phase 1 run from $RESUME_CKPT"
  INIT_ARGS=(--resume "$RESUME_CKPT")
else
  echo "[phase1-a100] Initializing fresh continual pretraining from $BASE_CKPT"
  INIT_ARGS=(--init-from "$BASE_CKPT")
fi

PYTHONPATH="$ROOT" python3 -u scripts/train.py \
  --config configs/kuon-1-loop.json \
  --data-dir "$DATA_DIR" \
  --steps "$STEPS" \
  --batch-size "$BS" \
  --grad-accum "$ACCUM" \
  --lr "$LR" \
  --min-lr "$MIN_LR" \
  --warmup-steps "$WARMUP" \
  --eval-interval "$EVAL_INT" \
  --checkpoint-dir "$CKPT_DIR" \
  --log-file "$LOG_FILE" \
  --compile \
  "${INIT_ARGS[@]}"

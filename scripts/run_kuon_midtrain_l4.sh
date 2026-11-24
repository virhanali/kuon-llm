#!/usr/bin/env bash
# Phase 1.5 Mid-Training / Annealing for Kuon 1 (119.61M Gated-Loop 24L)
# Continues from Phase 1 best base checkpoint (1.31B tokens, step 16,250, val 2.926)
# for +1.00B reasoning/QA/business/math bilingual tokens (total ~2.31B tokens).
#
# 100% resumable across Colab Pro L4 sessions:
#   - Verifies exact tokenizer SHA-256 and Phase 1 base checkpoint before burning GPU.
#   - First run uses --init-from (fresh AdamW + gentle cosine annealing 6e-5 -> 5e-6).
#   - Resumed runs auto-detect kuon-1-loop-last.pt and pass --resume (with RNG step advance).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to Phase 1 kuon-1-loop-best.pt (step 16,250)}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to packed 1B mid-training corpus dir}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/kuon/phase1_5-midtrain-1b}"

EXPECTED_SHA="75962384c24ad5837bd0c03f7d7122e7f5bd87ec0442fb61e01c060d649e9474"

python3 - "$BASE_CKPT" "$DATA_DIR" "$EXPECTED_SHA" <<'PY'
import json, sys, torch
from hashlib import sha256
from pathlib import Path

base_p, data_p, exp_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert base_p.is_file(), f"BASE_CKPT missing: {base_p}"
meta_p = data_p / "meta.json"
tok_p = data_p / "tokenizer.json"
assert meta_p.is_file() and tok_p.is_file(), f"Invalid DATA_DIR: {data_p}"
meta = json.loads(meta_p.read_text(encoding="utf-8"))
actual_sha = sha256(tok_p.read_bytes()).hexdigest()
assert actual_sha == exp_sha, f"Tokenizer SHA mismatch: {actual_sha} != {exp_sha}"
assert meta.get("language_mix") == {"en": 60, "id": 40}, f"Unexpected language_mix: {meta.get('language_mix')}"
for fn in ("train.en.bin", "train.id.bin", "val.en.bin", "val.id.bin"):
    fp = data_p / fn
    assert fp.is_file() and fp.stat().st_size > 100_000, f"Missing/tiny bin: {fp}"
st = torch.load(base_p, map_location="cpu", weights_only=False)
print(
    f"[midtrain-preflight] OK | base_step={st.get('step')} base_val={round(float(st.get('val_loss', 0)), 4)} "
    f"| corpus={meta.get('corpus')} | EN={meta['tokens']['en']['train']:,} ID={meta['tokens']['id']['train']:,}"
)
PY

# 60/40 bilingual requires grad_accum divisible by 5 -> 61,440 tok/step -> 16,280 steps = 1.00B tokens
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
LR="${LR:-6e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WARMUP="${WARMUP:-300}"
EVAL_INT="${EVAL_INT:-$DEFAULT_EVAL}"

CKPT_DIR="$OUT_DIR/checkpoints"
LOG_FILE="$OUT_DIR/logs/kuon-1-loop-midtrain-1b.jsonl"
mkdir -p "$CKPT_DIR" "$(dirname "$LOG_FILE")"

RESUME_CKPT="$CKPT_DIR/kuon-1-loop-last.pt"
INIT_ARGS=()
if [ -f "$RESUME_CKPT" ]; then
  echo "[midtrain] Resuming existing Phase 1.5 mid-training from $RESUME_CKPT"
  INIT_ARGS=(--resume "$RESUME_CKPT")
else
  echo "[midtrain] Initializing Phase 1.5 mid-training from $BASE_CKPT (LR=$LR -> $MIN_LR, steps=$STEPS)"
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

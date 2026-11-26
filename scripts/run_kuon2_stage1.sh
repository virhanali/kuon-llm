#!/usr/bin/env bash
# Stage 1 Pretraining for Kuon-2 (295.75M Dense 24L, 1024-dim, GQA 16:4, SwiGLU 2816).
#
# Maintains an IDENTICAL 81,920 tokens/step batch size across GPUs (grad_accum divisible by 5
# for exact 60/40 EN/ID bilingual mixing):
#   - Kaggle Free T4x2 DDP (2x 15GB) : BS=8,  ACCUM=5  -> 81,920 tok/step (--grad-checkpoint)
#   - Single T4 / L4 (15-24GB)       : BS=8,  ACCUM=10 -> 81,920 tok/step (--grad-checkpoint)
#   - Single A100 (40-80GB)          : BS=16, ACCUM=5  -> 81,920 tok/step
#
# 100% resumable across Kaggle 9h chunks or Colab sessions:
#   - If OUT_DIR/checkpoints/kuon-2-dense-last.pt (or RESUME_CKPT) exists -> passes --resume.
#   - Else if BASE_CKPT is set -> passes --init-from BASE_CKPT (fresh LR schedule on new corpus).
#   - Else starts fresh from random initialization (step 0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:?Set DATA_DIR to packed bilingual corpus dir (must contain meta.json & tokenizer.json)}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/kuon/kuon2-stage1}"
EXPECTED_SHA="75962384c24ad5837bd0c03f7d7122e7f5bd87ec0442fb61e01c060d649e9474"

python3 - "$DATA_DIR" "$EXPECTED_SHA" "${BASE_CKPT:-}" "${RESUME_CKPT:-}" <<'PY'
import json, sys
from hashlib import sha256
from pathlib import Path

data_p, exp_sha, base_p, resume_p = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
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
if base_p:
    assert Path(base_p).is_file(), f"BASE_CKPT missing: {base_p}"
if resume_p:
    assert Path(resume_p).is_file(), f"RESUME_CKPT missing: {resume_p}"
print(
    f"[kuon2-stage1-preflight] OK | corpus={meta.get('corpus')} "
    f"| EN={meta['tokens']['en']['train']:,} ID={meta['tokens']['id']['train']:,}"
)
PY

NGPU="$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)"

if [[ "${NGPU:-1}" -ge 2 ]]; then
  DEFAULT_BS=8
  DEFAULT_ACCUM=5
  DEFAULT_EVAL=250
  DEFAULT_GC=1
  DEFAULT_COMPILE=0
  LAUNCHER=(python3 -m torch.distributed.run --standalone "--nproc_per_node=${NGPU}")
elif [[ "${VRAM_MB:-0}" -ge 36000 ]]; then
  DEFAULT_BS=16
  DEFAULT_ACCUM=5
  DEFAULT_EVAL=500
  DEFAULT_GC=0
  DEFAULT_COMPILE=1
  LAUNCHER=(python3 -u)
elif [[ "${VRAM_MB:-0}" -ge 21000 ]]; then
  # 24GB L4: BS=4x20 = exact 81,920 tok/step, ~11GB VRAM without grad_checkpoint + native BF16 + torch.compile
  DEFAULT_BS=4
  DEFAULT_ACCUM=20
  DEFAULT_EVAL=250
  DEFAULT_GC=0
  DEFAULT_COMPILE=1
  LAUNCHER=(python3 -u)
else
  DEFAULT_BS=4
  DEFAULT_ACCUM=20
  DEFAULT_EVAL=250
  DEFAULT_GC=1
  DEFAULT_COMPILE=0
  LAUNCHER=(python3 -u)
fi

STEPS="${STEPS:-12210}"
BS="${BS:-$DEFAULT_BS}"
ACCUM="${ACCUM:-$DEFAULT_ACCUM}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP="${WARMUP:-500}"
EVAL_INT="${EVAL_INT:-$DEFAULT_EVAL}"
USE_GC="${USE_GC:-$DEFAULT_GC}"
USE_COMPILE="${USE_COMPILE:-$DEFAULT_COMPILE}"

CKPT_DIR="$OUT_DIR/checkpoints"
LOG_FILE="$OUT_DIR/logs/kuon-2-dense.jsonl"
mkdir -p "$CKPT_DIR" "$(dirname "$LOG_FILE")"

EXTRA_ARGS=()
LOCAL_LAST="$CKPT_DIR/kuon-2-dense-last.pt"
if [[ -f "$LOCAL_LAST" ]]; then
  echo "[kuon2-stage1] Auto-resuming from $LOCAL_LAST"
  EXTRA_ARGS+=(--resume "$LOCAL_LAST")
elif [[ -n "${RESUME_CKPT:-}" ]]; then
  echo "[kuon2-stage1] Resuming from RESUME_CKPT=$RESUME_CKPT"
  EXTRA_ARGS+=(--resume "$RESUME_CKPT")
elif [[ -n "${BASE_CKPT:-}" ]]; then
  echo "[kuon2-stage1] Continual pretraining (--init-from) from BASE_CKPT=$BASE_CKPT"
  EXTRA_ARGS+=(--init-from "$BASE_CKPT")
else
  echo "[kuon2-stage1] Fresh Stage 1 pretraining from step 0"
fi

if [[ -n "${STOP_STEP:-}" ]]; then
  EXTRA_ARGS+=(--stop-step "$STOP_STEP")
fi
if [[ -n "${MAX_SECONDS:-}" ]]; then
  EXTRA_ARGS+=(--max-seconds "$MAX_SECONDS")
fi
if [[ "${USE_GC}" == "1" ]]; then
  EXTRA_ARGS+=(--grad-checkpoint)
else
  EXTRA_ARGS+=(--no-grad-checkpoint)
fi
if [[ "${USE_COMPILE}" == "1" ]]; then
  EXTRA_ARGS+=(--compile)
fi

echo "[kuon2-stage1] NGPU=$NGPU VRAM=${VRAM_MB}MB BS=$BS ACCUM=$ACCUM STEPS=$STEPS LR=$LR->$MIN_LR GC=$USE_GC"

PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU \
  "${LAUNCHER[@]}" scripts/train.py \
  --config configs/kuon-2-dense.json \
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
  "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
# Stage 2 Mid-Training / Gated-Loop Upcycling for Kuon-2 (295.81M params, 24 stored -> 32 effective layers).
#
# Zero-loss upcycles Stage 1 dense checkpoint (`kuon-2-dense-best.pt`) into `configs/kuon-2-loop.json`
# and trains for +1.00B reasoning/QA/math/business tokens (`virhanali/kuon-midtrain-1b`, 12,210 steps
# at 81,920 tokens/step) with gentle cosine annealing (6e-5 -> 5e-6) and activation checkpointing.
#
# 100% resumable across Colab Pro L4 / A100 sessions:
#   - First run uses --upcycle-from BASE_CKPT (verifies dense 24L 1024d checkpoint).
#   - Subsequent runs auto-detect OUT_DIR/checkpoints/kuon-2-loop-last.pt and pass --resume.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to Stage 1 kuon-2-dense-best.pt}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to packed 1B mid-training corpus dir (kuon-midtrain-1b)}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/kuon/kuon2-stage2-loop-midtrain}"
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
mc = st["model_cfg"]
assert mc["n_layer"] == 24 and mc["n_embd"] == 1024, f"Unexpected base model_cfg: {mc}"
print(
    f"[kuon2-midtrain-preflight] OK | base_step={st.get('step')} base_val={round(float(st.get('val_loss', 0)), 4)} "
    f"| corpus={meta.get('corpus')} | EN={meta['tokens']['en']['train']:,} ID={meta['tokens']['id']['train']:,}"
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
elif [[ "${VRAM_MB:-0}" -ge 72000 ]]; then
  # 80GB A100/H100: 32L effective fits comfortably at BS=16x5 without grad_checkpoint
  DEFAULT_BS=16
  DEFAULT_ACCUM=5
  DEFAULT_EVAL=500
  DEFAULT_GC=0
  DEFAULT_COMPILE=1
  LAUNCHER=(python3 -u)
elif [[ "${VRAM_MB:-0}" -ge 36000 ]]; then
  # 40GB A100: 32L effective at BS=8x10 keeps exact 81,920 tok/step safely under ~20GB VRAM with torch.compile
  DEFAULT_BS=8
  DEFAULT_ACCUM=10
  DEFAULT_EVAL=500
  DEFAULT_GC=0
  DEFAULT_COMPILE=1
  LAUNCHER=(python3 -u)
elif [[ "${VRAM_MB:-0}" -ge 21000 ]]; then
  # 24GB L4: 32L effective at BS=4x20 uses ~13GB VRAM without grad_checkpoint + native BF16 + torch.compile
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
LR="${LR:-6e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WARMUP="${WARMUP:-300}"
EVAL_INT="${EVAL_INT:-$DEFAULT_EVAL}"
USE_GC="${USE_GC:-$DEFAULT_GC}"
USE_COMPILE="${USE_COMPILE:-$DEFAULT_COMPILE}"

CKPT_DIR="$OUT_DIR/checkpoints"
LOG_FILE="$OUT_DIR/logs/kuon-2-loop-midtrain.jsonl"
mkdir -p "$CKPT_DIR" "$(dirname "$LOG_FILE")"

EXTRA_ARGS=()
if [[ "${USE_GC}" == "1" ]]; then
  EXTRA_ARGS+=(--grad-checkpoint)
else
  EXTRA_ARGS+=(--no-grad-checkpoint)
fi
LOCAL_LAST="$CKPT_DIR/kuon-2-loop-last.pt"
if [[ -f "$LOCAL_LAST" ]]; then
  echo "[kuon2-midtrain] Auto-resuming Stage 2 loop mid-training from $LOCAL_LAST"
  EXTRA_ARGS+=(--resume "$LOCAL_LAST")
elif [[ -n "${RESUME_CKPT:-}" ]]; then
  echo "[kuon2-midtrain] Resuming Stage 2 loop mid-training from RESUME_CKPT=$RESUME_CKPT"
  EXTRA_ARGS+=(--resume "$RESUME_CKPT")
else
  echo "[kuon2-midtrain] Zero-loss upcycling Stage 1 dense checkpoint $BASE_CKPT -> 32L Gated-Loop"
  EXTRA_ARGS+=(--upcycle-from "$BASE_CKPT")
fi

if [[ -n "${STOP_STEP:-}" ]]; then
  EXTRA_ARGS+=(--stop-step "$STOP_STEP")
fi
if [[ -n "${MAX_SECONDS:-}" ]]; then
  EXTRA_ARGS+=(--max-seconds "$MAX_SECONDS")
fi
if [[ "${USE_COMPILE}" == "1" ]]; then
  EXTRA_ARGS+=(--compile)
fi

echo "[kuon2-midtrain] NGPU=$NGPU VRAM=${VRAM_MB}MB BS=$BS ACCUM=$ACCUM STEPS=$STEPS LR=$LR->$MIN_LR"

PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU \
  "${LAUNCHER[@]}" scripts/train.py \
  --config configs/kuon-2-loop.json \
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

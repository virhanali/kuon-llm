#!/usr/bin/env bash
# Kaggle T4 pipeline: Cosmopedia -> 350M-token bins -> kuon-113m pretrain.
# Usage (inside a Kaggle notebook console, after cloning the repo):
#   bash scripts/kaggle_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/pack_corpus.py \
  --input /kaggle/working/cosmopedia.txt \
  --tokenizer tokenizers/cosmopedia-16k.json \
  --out data/cosmopedia \
  --max-tokens 350000000 \
  --workers 2

python scripts/train.py --config configs/kuon-113m.json --steps 4500

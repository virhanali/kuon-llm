#!/usr/bin/env bash
# Start Kuon Studio with the Kuon-13M SFT chat model as the engine.
# Usage (from the repo root or anywhere):  bash deploy/run.sh
set -e
cd "$(dirname "$0")/.."
exec venv/bin/python -m kuon_agent.server \
    --engine kuon \
    --checkpoint checkpoints/kuon-13m-sft.pt \
    --bind 127.0.0.1 \
    --port 8000

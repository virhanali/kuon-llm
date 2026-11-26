# Kuon

Kuon is a small, open-source language model and evidence-grounded business agent built from scratch in PyTorch. The repository contains the complete stack with zero external LLM framework wrappers: byte-level BPE tokenizer, Gated-Loop Transformer, distributed pretraining and mid-training, supervised fine-tuning (SFT), preference alignment (Context-APO-zero / ORPO), deterministic analytical tools, holdout evaluation suites, and an OpenAI-compatible HTTP server.

> **Research software notice:** Kuon is an experimental small-model research project focused on English–Indonesian sales, customer service, and operations tasks. Its outputs can still be wrong or incomplete and should be verified before making business decisions.

## Model configurations

| Config | Parameters | Stored / Effective Depth | Context | Vocab | Notes |
|---|---:|---|---:|---:|---|
| [`configs/kuon-13m.json`](configs/kuon-13m.json) | 12.59M | 6 / 6 layers | 512 | 8,192 | Lightweight demo ([public demo weights](https://github.com/virhanali/kuon-llm/releases/tag/v0.3-sft)) |
| [`configs/kuon-1-loop.json`](configs/kuon-1-loop.json) | 119.61M | 16 / 24 layers | 1,024 | 24,576 | Earlier 120M Gated-Loop prototype |
| [`configs/kuon-2-dense.json`](configs/kuon-2-dense.json) | 295.75M | 24 / 24 layers | 1,024 | 24,576 | Kuon 1 Stage 1 dense pretraining base (`2.40B` tokens) |
| [`configs/kuon-2-loop.json`](configs/kuon-2-loop.json) | 295.81M | 24 / 32 layers | 1,024 | 24,576 | **Kuon 1** release architecture (`3.40B` tokens + Stage 3 v4.2, `307 MB` INT8) |

### Architecture highlights (Kuon 1 — `configs/kuon-2-loop.json`)

- **Recurrent depth (Gated-Loop):** 24 stored transformer blocks where the 8 middle blocks (`8..15`) execute twice via a learned low-rank residual gate (`loop_gate_rank=32`), giving 32 effective layers with negligible parameter overhead.
- **Modern transformer core:** Pre-norm RMSNorm, RoPE positional embeddings, Grouped-Query Attention (12 query heads / 4 KV heads), SwiGLU feed-forward blocks, tied input/output embeddings, and per-pass KV caching for generation.
- **Balanced bilingual tokenizer:** 24,576-token byte-level BPE (`tokenizers/kuon1-bilingual-24k.json`) trained on balanced English–Indonesian text and business/numeric formats.
- **CPU-friendly INT8 weights:** Portable `307 MB` INT8 release checkpoint (`kuon/quant.py`, `3.99x` smaller than FP32) running at ~`13.9 tok/s` on 4 CPU threads in lossless `int8_weights` mode with `100%` FP32 benchmark parity.

### Kuon 1 release evaluation (Stage 3 v4.2, FP32 & INT8 identical)

| Metric | Target | Kuon 1 INT8 (`307 MB`) |
|---|---:|---:|
| Grounded calibration accuracy (`16` unseen EN/ID cases) | High | `15/16` (`93.8%`: EN `8/8`, ID `7/8`) |
| Unanswerable restraint | `>= 90%` | `100.0%` (`8/8` EN & ID) |
| Answerable recall | `>= 75%` | `87.5%` (`7/8`: EN `100%`, ID `75%`) |
| Hallucination rate | `< 10%` | `6.2%` (`1/16`, EN `0.0%`) |
| Pillar 2 business deliverables rubric | `>= 90%` | `94.4%` (`34/36`) |
| Confirmed causal action selection | Stable | `59.7%` (`43/72`) |

## Quickstart

Requires Python 3.10+ and PyTorch 2.4+.

```bash
git clone https://github.com/virhanali/kuon-llm.git
cd kuon-llm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Verify installation
pytest -q
```

### Run the public 13M demo on CPU

```bash
mkdir -p checkpoints/kuon-13m
curl -fL https://github.com/virhanali/kuon-llm/releases/download/v0.3-sft/kuon-13m-sft-v4.pt \
    -o checkpoints/kuon-13m/kuon-13m-sft-v4.pt
curl -fL https://github.com/virhanali/kuon-llm/releases/download/v0.1-sft/tokenizer.json \
    -o checkpoints/kuon-13m/tokenizer.json

python scripts/generate.py \
    --checkpoint checkpoints/kuon-13m/kuon-13m-sft-v4.pt \
    --tokenizer checkpoints/kuon-13m/tokenizer.json \
    --prompt "$(printf 'User: Hello\nAssistant:')" \
    --temperature 0 --tokens 40
```

Always pair a checkpoint with its matching tokenizer and only load `.pt` files from trusted sources.

## OpenAI-compatible API and business agent

`kuon_agent.server` serves both a single-page chat UI (`/`) and an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/health`) using Python's standard library.

```bash
python -m kuon_agent.server \
    --engine kuon \
    --checkpoint path/to/kuon-inference.pt \
    --no-wiki
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "kuon-agent",
      "x_kuon_mode": "business",
      "messages": [
        {"role": "user", "content": "| Week | Revenue |\n|---|---|\n| W1 | 20000 |\n| W2 | 15000 |\nWhat is the revenue percentage change from W1 to W2?"}
      ]
    }'
```

In business mode (`x_kuon_mode: "business"`), the agent combines the model with deterministic verification primitives in `kuon_agent/tools.py` and `kuon_agent/agent.py`:

- **Safe arithmetic & KPI formulas:** AST-verified calculator (`safe_calc`) and exact formulas (`business_kpi`) for gross/net margin, percentage change, conversion rate, AOV, break-even units, ROI, and EOQ.
- **Table & multi-turn analytics:** Parses Markdown, CSV, and Excel/Google Sheets TSV tables (`summarize_table`), computes exact period deltas, highest/lowest, totals, averages, and percentage shares, handles multi-turn missing values or corrections, and checks currency consistency.
- **Hybrid RAG (MIT):** Pass `--rag path/to/docs --semantic-rag` (or `KUON_SEMANTIC_RAG=1`) to combine BM25 with `intfloat/multilingual-e5-small` ONNX (`118 MB`, ~`6.3 ms` CPU query latency) across `.txt`, `.md`, `.csv`, and `.tsv` files, with automatic multi-turn follow-up query expansion.
- **Live web grounding (General QA only):** Set `EXA_API_KEY` in `.env` or environment secrets to enable Exa AI live highlights with publication dates, trusted-domain priority, multi-domain diversity, and post-generation number/currency verification. Business mode stays isolated from the public web.
- **Production HTTP server:** Supports live SSE token streaming (`stream: true`), serialized inference queue telemetry in `/health`, optional Bearer authentication via `KUON_API_KEY`, 1 MB body cap, and `x_notes` / `x_sources` provenance fields.

See [`deploy/instabox-setup.sh`](deploy/instabox-setup.sh) and [`deploy/README.md`](deploy/README.md) for CPU-only VPS / InstaBox deployment (`4 vCPU / 8 GiB RAM`).

## Training pipeline

Kuon 1 follows a four-stage, fully reproducible training pipeline:

```bash
# 1. Pack bilingual pretraining corpus (60% EN / 40% ID) and pretrain
python scripts/prepare_kuon1_corpus.py \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --out data/kuon1-pretrain --train-tokens 1000000000

python scripts/train.py \
    --config configs/kuon-1-loop.json \
    --data-dir data/kuon1-pretrain \
    --compile

# 2. Pack disjoint reasoning/QA/business mid-training corpus and anneal
python scripts/prepare_kuon1_midtrain.py \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --out data/kuon1-midtrain --train-tokens 1000000000

BASE_CKPT=checkpoints/kuon-1-loop/kuon-1-loop-best.pt \
DATA_DIR=data/kuon1-midtrain \
OUT_DIR=checkpoints/kuon-1-midtrain \
bash scripts/run_kuon_midtrain_l4.sh

# 3. Stage 3A: Supervised fine-tuning (SFT)
python scripts/prepare_kuon1_sft.py \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --out data/kuon1-sft

python scripts/sft.py \
    --base checkpoints/kuon-1-midtrain/checkpoints/kuon-1-loop-best.pt \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --data data/kuon1-sft \
    --out checkpoints/kuon-1-sft.pt

# 4. Stage 3B: Preference alignment (Context-APO-zero or ORPO)
python scripts/prepare_kuon1_prefs.py \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --out data/kuon1-prefs

python scripts/align.py \
    --base checkpoints/kuon-1-sft.pt \
    --tokenizer tokenizers/kuon1-bilingual-24k.json \
    --data data/kuon1-prefs \
    --out checkpoints/kuon-1-aligned.pt \
    --method apo_zero
```

For end-to-end Stage 3 alignment + evaluation + inference export in one command, see `notebooks/kuon1_stage3_align_runner.py`.

### Lossless inference export

Strip optimizer state from a training checkpoint without altering model weights:

```bash
python scripts/export_inference.py \
    --checkpoint path/to/training.pt \
    --out path/to/inference.pt
```

## Evaluation suites

Holdout evaluation scripts are kept strictly disjoint from training splits:

| Script | What it measures |
|---|---|
| `scripts/eval_business_agent.py` | Bilingual grounded calibration (recall, restraint, hallucination rate) and KPI/table deliverables |
| `scripts/eval_confirmed_actions.py` | Fact-grounded causal action selection (A/B/C symmetry and permutation checks) |
| `scripts/eval_action_choice.py` | Sensitivity to flipped operational facts across EN/ID business scenarios |
| `scripts/eval_calibration.py` | Local business calibration + public AA-Omniscience v1.1 factual restraint probe |
| `scripts/eval_gdpval.py` | Open-ended business deliverable prompts (keyword/rubric proxy) |
| `scripts/eval_chat.py` | Qualitative EN/ID business chat inspection |

Note: the local AA-Omniscience and GDPval scripts use lightweight heuristic scoring and are intended for internal iteration rather than official leaderboard comparisons.

## Repository layout

| Path | Description |
|---|---|
| `kuon/` | Core library: `tokenizer.py`, `model.py`, `data.py`, `trainer.py`, `sft.py`, `align.py`, `generate.py`, `chat.py`, `config.py` |
| `kuon_agent/` | Agent loop (`agent.py`), deterministic tools (`tools.py`), inference engines (`llm.py`), BM25 RAG (`rag.py`), OpenAI server (`server.py`) |
| `configs/` | Model & training configurations (`kuon-13m.json`, `kuon-1-dense.json`, `kuon-1-loop.json`) |
| `datasets/business/` | Curated EN/ID business training and holdout evaluation sets |
| `tokenizers/` | Trained byte-level BPE tokenizer definitions |
| `scripts/` | Corpus builders, training/alignment entry points, inference export, and evaluation suites |
| `notebooks/` | End-to-end Stage 3 alignment & evaluation runner |
| `deploy/` | Dockerfile, Caddyfile, systemd unit, and hosting notes |
| `tests/` | Unit and end-to-end integration test suite |

## License

Code in this repository is licensed under the [MIT License](LICENSE). External datasets, dependencies, and model weights may carry separate licensing and attribution terms—verify them before commercial use or redistribution.

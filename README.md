# Kuon

A small, English-first, open-weight LLM built **from scratch** — tokenizer,
transformer, training loop, and inference — in a codebase small enough to
read end-to-end in an afternoon.

Kuon's architecture is the modern standard (Llama / Qwen / Mistral lineage):

| component | choice |
|---|---|
| normalization | RMSNorm (pre-norm) |
| positions | RoPE (rotary embeddings) |
| attention | grouped-query attention (GQA) + KV cache |
| feed-forward | SwiGLU |
| embeddings | tied (lm_head = token embedding) |
| tokenizer | byte-level BPE (GPT-2 pre-tokenizer) |
| objective | next-token prediction (cross-entropy) |

Every line of the stack lives in this repo: no black-box libraries between
you and the model. Read it in order and you will understand how a modern
LLM is actually built.

## Quickstart

```bash
make setup           # venv + install (uses uv)
make test            # unit + end-to-end tests
make data            # download TinyStories, train tokenizer, pack tokens
make train           # pretrain kuon-13m
make generate        # sample text from the checkpoint
```

## The code, in reading order

| # | file | what you will understand |
|---|---|---|
| 1 | `kuon/config.py` | what a "model size" actually is |
| 2 | `kuon/tokenizer.py` | byte-level BPE: how text becomes integers |
| 3 | `kuon/data.py` | packed token streams and next-token batches |
| 4 | `kuon/model.py` | the transformer, bottom-up: RMSNorm → RoPE → KVCache → GQA attention → SwiGLU → blocks |
| 5 | `kuon/generate.py` | temperature, top-k, top-p, and cached decoding |
| 6 | `kuon/trainer.py` | AdamW, cosine LR, gradient accumulation, checkpoints |
| 7 | `scripts/` | how everything is wired: `prepare_data`, `train`, `generate`, `info` |

## Models

| config | params | context | intended device |
|---|---|---|---|
| `kuon-13m` | ~12.6M | 512 | laptop CPU (overnight smoke runs) |
| `kuon-48m` | ~48.4M | 1024 | free Colab / Kaggle T4 |
| `kuon-107m` | ~107M | 1024 | Colab T4 16GB |

Exact numbers: `python scripts/info.py --config configs/kuon-107m.json`

## Hardware reality

Everything in this repo runs on a laptop CPU: tests in a couple of minutes,
data prep, and meaningful smoke training of `kuon-13m`. Real pretraining
(billions of tokens for 48m/107m) is sized for free cloud GPUs, and the
repo is identical there:

```bash
# in Colab / Kaggle, attached to this folder:
pip install -e .
python scripts/prepare_data.py --source tinystories --out data/tinystories
python scripts/train.py --config configs/kuon-107m.json --device cuda
```

## Data

`scripts/prepare_data.py` streams a corpus, optionally trains the tokenizer
on a sample, then packs every document into `train.bin` / `val.bin` as
uint16 token ids (documents separated by `<endoftext>`), memory-mapped at
training time. Sources: `tinystories` (Hugging Face) or any text `file`.

## Experiments

- **[Looped Kuon](docs/experiments/looped.md)** — recurrent depth (middle
  blocks executed multiple times, Inkling 2 Mini style). Config:
  `configs/kuon-13m-loop.json`. Compare against the dense baseline at equal
  tokens and at equal FLOPs.

## SFT (chat)

Fine-tune a pretrained checkpoint on `{user, assistant}` pairs; the loss is
masked to the assistant turns only (`kuon/chat.py`, `kuon/sft.py`):

```bash
python scripts/prepare_sft.py --corpus corpus.txt --out data/sft --n 3000
python scripts/sft.py --base checkpoints/kuon-13m/kuon-13m-last.pt \
    --data data/sft --out checkpoints/kuon-13m-sft.pt --steps 300
```

## Kuon Agent (from scratch)

A `search → think → answer` loop with **no agent framework**, living next to
the model in `kuon_agent/`:

- `rag.py` — BM25 retrieval implemented from scratch (inverted index, idf,
  length normalization); index local `.txt`/`.md` corpora
- `search.py` — Wikipedia API + DuckDuckGo HTML endpoints, stdlib only
- `agent.py` — gathers evidence, runs an explicit *think* step (bullet
  notes), then answers **only from evidence**, with `[n]` citations
- `llm.py` — swappable engines: Kuon checkpoint, Ollama, llama.cpp, mock

```bash
python -m kuon_agent.cli "who is mr beast"                        # Wikipedia + mock engine
python -m kuon_agent.cli "who is mr beast" --engine ollama --model qwen2.5-coder:7b
python -m kuon_agent.cli "what is our launch date" --rag notes/ --no-wiki
```

The agent never answers without evidence: when search/RAG returns nothing it
says so. Swapping the engine changes answer quality; the loop and the
grounding stay identical.

## Roadmap

- [x] tokenizer, model, trainer, inference, tests — all from scratch
- [x] pretrain `kuon-13m` on TinyStories — Kaggle T4, 2000 steps, val 2.35
- [x] looped-transformer experiment harness + SFT pipeline (+ tests)
- [ ] run the looped experiment and publish the comparison
- [ ] SFT run: `kuon-13m` that answers chat prompts
- [ ] pretrain `kuon-48m` / `kuon-107m` (Colab / Kaggle GPU)
- [ ] preference tuning (DPO)
- [ ] export to GGUF + llama.cpp for quantized local inference
- [x] from-scratch agent: Wikipedia/DuckDuckGo search + BM25 RAG + grounded answering
- [ ] run the agent with a local open model (Ollama) and with Kuon after SFT


## License

MIT

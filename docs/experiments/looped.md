# Experiment: Looped Kuon (recurrent depth)

**Status:** designed, ready to run
**Inspired by:** Thinking Machines, *Inkling 2 Mini* (looped transformer,
Nov 2026) and the universal-transformer / recurrent-depth line of work.

## Hypothesis

Repeating the **middle** transformer blocks of Kuon-13m — reusing the same
weights, no extra parameters — gives better loss per unit of compute than
spending that compute on a deeper dense model.

Inkling 2 Mini's ablation found a compute-equivalent gain of **1.51–1.91×**
for looping the middle 12 of 24 layers, versus 1.24–1.36× for the last 12.
At 13M parameters on TinyStories we do not expect the same magnitude; the
question is whether the direction holds.

## Design

| Run | Config | Block applications / token | Stored layers |
|---|---|---|---|
| A — baseline | `configs/kuon-13m.json` (steps=1000) | 6 | 6 |
| B — looped | `configs/kuon-13m-loop.json` | 10 (middle 4 × 2 passes + 2) | 6 |

Both runs: same data (`data/tinystories`, 200k-story corpus), same token
budget (1000 steps × 16,384 tokens = 16.4M tokens), same optimizer and
schedule. Run B uses **~1.67× the FLOPs per token** for the same parameter
count — so compare:

1. **Val loss at equal tokens** (steps 100 … 1000) — B should be lower if
   looping helps at all;
2. **Val loss per estimated FLOPs** — the honest compute-normalized view;
3. **Sample quality** at step 1000 (`sample>` lines in the training log).

Kuon-13m already has a completed 2000-step reference run
(`logs/kuon-13m.jsonl` in the Kaggle notebook); for the controlled
comparison, rerun A with `--steps 1000` on the same corpus/seed.

## How to run (Kaggle T4 console)

```bash
!cd /kaggle/working/repo && PYTHONPATH=$PWD python scripts/train.py --config configs/kuon-13m.json --steps 1000
!cd /kaggle/working/repo && PYTHONPATH=$PWD python scripts/train.py --config configs/kuon-13m-loop.json
```

Each run takes ~10–15 minutes on a T4. Results go to
`logs/kuon-13m.jsonl` and `logs/kuon-13m-loop.jsonl`.

## What we deliberately did NOT implement (yet)

Inkling 2 Mini adds two ingredients that make looping train stably at scale:

1. **Gated message**: pass 2+ reads the output of the previous pass through a
   learned gate that starts closed (so training begins identical to a
   non-looped model) and opens up over training.
2. **Attention residuals** (per Kimi et al.): every layer learns its own mix
   of earlier block outputs instead of a plain running sum.

Our first experiment is the minimal version: plain residual stream, middle
blocks repeated. If the direction holds, the gate is the natural next step
(a small, self-contained change in `Block.forward`).

## Results

_Fill after running. Keep the raw JSONL logs with the run._

| Step | A val (dense 6L) | B val (loop 6L, mid×2) | Notes |
|---|---|---|---|
| 100 | | | |
| 500 | | | |
| 1000 | | | |

## Caveats

- One corpus, one seed, one scale — evidence, not proof.
- Looping increases FLOPs per token; a "win" at equal tokens is partly a
  compute win. Report both views.
- Generating from looped models currently falls back to no-cache decoding
  (`kuon/generate.py`), which is fine at this scale.

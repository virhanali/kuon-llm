# Kuon Business Datasets

Curated English and Indonesian business datasets used by Kuon 1 for grounded
reading, abstention, and counterfactual operational advice. All scenarios are
fictional and contain no real customer records.

## Files

| File | Purpose |
|---|---|
| `train.jsonl` | Seed grounded business QA and abstention examples |
| `eval.jsonl` | Holdout business evaluation questions (never include in training) |
| `scenarios.tsv` | 54 fictional EN/ID operational scenarios expanded by `scripts/prepare_balanced_business.py` |
| `advice_pairs.txt` | 60 paired EN/ID causal advice scenarios (each with opposing follow-up facts) expanded by `scripts/prepare_advice_pairs.py` |
| `advice_train.jsonl` | Multi-turn conversational business advice seed rows |
| `advice_eval.jsonl` | Holdout conversational advice evaluation cases (never include in training) |

## Schema and evaluation rules

- Rows use `user`, `assistant`, and an optional `evidence` field containing numbered excerpts (`[1] ...`).
- Keep `eval.jsonl` and `advice_eval.jsonl` strictly disjoint from every pretraining, mid-training, SFT, and preference split.
- Generate the expanded scenario splits locally with:

```bash
python scripts/prepare_balanced_business.py --out data/business-balanced
python scripts/prepare_advice_pairs.py --out data/advice-pairs
```

## Licensing

Original Kuon business scenarios and evaluation rubrics in this directory are
released under the repository's MIT License. External public datasets combined
during full SFT/mid-training (`SmolTalk`, `Aya`, `SQuAD v2`, `DROP`, `HotpotQA`,
`GSM8K`, `FinQA`, `TyDiQA`, `PleIAs/SYNTH`, `FineMath`, `FineWeb-Edu`,
`FineWeb2-HQ`) retain their respective upstream licenses.

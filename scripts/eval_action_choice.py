"""Evaluate whether Kuon selects different actions when user facts change.

Tests both:
1. `mc_letter`: Multiple-choice A/B/C with randomized option positions (scores
   next-token log-probabilities for " A", " B", " C").
2. `completion_ll`: Direct length-normalized log-likelihood of each candidate
   action under Kuon's standard chat prompt format (no A/B/C format required).

A scenario pair (Fact A, Fact B) counts as PASS only if Fact A selects Action A
AND Fact B selects Action B.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from kuon.chat import format_prompt
from kuon.config import ModelConfig
from kuon.model import Transformer
from kuon.tokenizer import BPETokenizer


CLARIFY_LABEL = {
    "id": "Tanya dulu untuk memastikan penyebab utamanya sebelum mengambil tindakan.",
    "en": "Ask one clarifying question first before choosing an action.",
}

# Built-in action-choice labels for the two paired scenarios in advice_eval.jsonl
# (used as comparison benchmark only, never for training).
HOLDOUT_PAIRS = [
    {
        "scenario": "id:holdout_bakery",
        "language": "id",
        "opening": "Toko kue saya makin sedikit yang beli.",
        "question": "Yang turun jumlah orang yang datang, atau jumlah pengunjung yang jadi membeli?",
        "fact_a": "Jumlah pengunjung turun setelah jalan depan toko ditutup. Pengunjung yang datang masih sering membeli.",
        "action_a": "Perbaiki akses jalan alternatif atau sediakan cara memesan dari jauh.",
        "fact_b": "Pengunjung sama banyak, tapi lebih sedikit yang beli. Banyak yang datang ketika pilihan kue sudah habis.",
        "action_b": "Tambah ketersediaan kue di jam ramai dan catat penjualan yang hilang karena stok habis.",
    },
    {
        "scenario": "en:holdout_class",
        "language": "en",
        "opening": "Fewer customers sign up for our classes.",
        "question": "Where do people stop: before viewing the timetable or while booking?",
        "fact_a": "Most people never reach the timetable. They often ask where to find it.",
        "action_a": "Make the timetable easier to find and measure timetable visits.",
        "fact_b": "Timetable views are steady. Customers report that the checkout shows an error after they choose a class.",
        "action_b": "Reproduce and fix the checkout error, then measure completed bookings.",
    },
]


def first_sentence(text: str) -> str:
    text = text.strip()
    for sep in (". ", "? ", "! "):
        idx = text.find(sep)
        if idx != -1:
            return text[: idx + 1].strip()
    return text


def load_dev_scenarios(path: Path) -> list[dict]:
    """Group advice_dev.jsonl (3 rows per scenario) into paired scenario dicts."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        grouped[row["scenario"]].append(row)

    scenarios = []
    for name, rows in grouped.items():
        clar = [r for r in rows if r["task"] == "clarification"]
        adapt = [r for r in rows if r["task"] == "adaptation"]
        if len(clar) != 1 or len(adapt) != 2:
            raise ValueError(f"Scenario {name} expected 1 clarification + 2 adaptation rows")
        lang = clar[0]["language"]
        opening = clar[0]["user"].strip()
        question = clar[0]["assistant"].strip()
        facts, actions = [], []
        for r in adapt:
            parts = r["user"].split("\nUser: ")
            fact = parts[-1].strip() if len(parts) > 1 else r["user"].strip()
            facts.append(fact)
            actions.append(first_sentence(r["assistant"]))
        scenarios.append(
            {
                "scenario": name,
                "language": lang,
                "opening": opening,
                "question": question,
                "fact_a": facts[0],
                "action_a": actions[0],
                "fact_b": facts[1],
                "action_b": actions[1],
            }
        )
    return scenarios


def build_items(scenarios: list[dict], seed: int = 1337) -> list[dict]:
    """Expand each scenario into 3 choice items with randomized option order."""
    rng = random.Random(seed)
    items = []
    for sc in scenarios:
        lang = sc["language"]
        clarify_text = CLARIFY_LABEL[lang]
        base_options = [
            ("clarify", clarify_text),
            ("action_a", sc["action_a"]),
            ("action_b", sc["action_b"]),
        ]
        cases = [
            ("clarification", sc["opening"], "clarify"),
            (
                "fact_a",
                f"{sc['opening']}\nAssistant: {sc['question']}\nUser: {sc['fact_a']}",
                "action_a",
            ),
            (
                "fact_b",
                f"{sc['opening']}\nAssistant: {sc['question']}\nUser: {sc['fact_b']}",
                "action_b",
            ),
        ]
        for case_kind, user_text, gold_key in cases:
            opts = list(base_options)
            rng.shuffle(opts)
            letters = ["A", "B", "C"]
            letter_to_key = {l: k for l, (k, _) in zip(letters, opts)}
            gold_letter = next(l for l, k in letter_to_key.items() if k == gold_key)
            items.append(
                {
                    "scenario": sc["scenario"],
                    "language": lang,
                    "case": case_kind,
                    "user": user_text,
                    "gold_key": gold_key,
                    "gold_letter": gold_letter,
                    "options": [(l, k, txt) for l, (k, txt) in zip(letters, opts)],
                    "letter_to_key": letter_to_key,
                }
            )
    return items


def format_mc_prompt(item: dict, *, business: bool = True) -> str:
    lang = item["language"]
    if lang == "id":
        header = "Pilih satu tindakan terbaik (A, B, atau C) berdasarkan fakta pengguna."
        cue = "Pilihan:"
    else:
        header = "Choose the single best action (A, B, or C) based on the user's facts."
        cue = "Choice:"
    lines = [f"{l}. {txt}" for l, _, txt in item["options"]]
    body = f"{item['user']}\n\n{header}\n" + "\n".join(lines) + f"\n{cue}"
    return format_prompt(body, business=business)


@torch.no_grad()
def score_mc_letter(model: Transformer, tokenizer: BPETokenizer, item: dict, device: str,
                    *, business: bool = True) -> dict:
    prompt = format_mc_prompt(item, business=business)
    ids = tokenizer.encode(prompt)
    if len(ids) > model.cfg.block_size:
        ids = ids[-model.cfg.block_size :]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)

    scores: dict[str, float] = {}
    for letter in ("A", "B", "C"):
        # Prefer leading-space token (" A"), fall back to bare ("A") if multi-token.
        toks = tokenizer.encode(" " + letter)
        if len(toks) != 1:
            toks = tokenizer.encode(letter)
        scores[letter] = log_probs[toks[0]].item()

    pred_letter = max(scores, key=scores.get)
    pred_key = item["letter_to_key"][pred_letter]
    return {
        "pred_letter": pred_letter,
        "pred_key": pred_key,
        "correct": pred_key == item["gold_key"],
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


@torch.no_grad()
def completion_mean_logprob(model: Transformer, tokenizer: BPETokenizer,
                            prompt: str, completion: str, device: str) -> float:
    p_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(f"{prompt} {completion.strip()}")
    if len(full_ids) > model.cfg.block_size + 1:
        overflow = len(full_ids) - (model.cfg.block_size + 1)
        p_ids = p_ids[overflow:]
        full_ids = full_ids[overflow:]
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    start = max(0, len(p_ids) - 1)
    comp_logits = logits[0, start:].float()
    comp_targets = y[0, start:]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_lp = log_probs.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)
    return token_lp.mean().item()


def score_completion_ll(model: Transformer, tokenizer: BPETokenizer, item: dict, device: str,
                        *, business: bool = True) -> dict:
    prompt = format_prompt(item["user"], business=business)
    scores: dict[str, float] = {}
    for _, key, txt in item["options"]:
        scores[key] = completion_mean_logprob(model, tokenizer, prompt, txt, device)
    pred_key = max(scores, key=scores.get)
    # 2-way contrast (action_a vs action_b only, ignoring clarify)
    ab_pred = "action_a" if scores["action_a"] >= scores["action_b"] else "action_b"
    return {
        "pred_key": pred_key,
        "correct": pred_key == item["gold_key"],
        "ab_pred": ab_pred,
        "ab_correct": (ab_pred == item["gold_key"]) if item["gold_key"] in ("action_a", "action_b") else None,
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def summarize_results(records: list[dict], mode: str) -> dict:
    by_sc: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_sc[r["scenario"]][r["case"]] = r

    total_items = len(records)
    correct_items = sum(1 for r in records if r[mode]["correct"])
    clar_rows = [r for r in records if r["case"] == "clarification"]
    adapt_rows = [r for r in records if r["case"] in ("fact_a", "fact_b")]

    pair_pass = 0
    pair_changed = 0
    ab_pair_pass = 0
    full3_pass = 0
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "pair_pass": 0, "ab_pair_pass": 0, "changed": 0})

    for sc, cases in by_sc.items():
        lang = cases["fact_a"]["language"]
        by_lang[lang]["pairs"] += 1
        a_res = cases["fact_a"][mode]
        b_res = cases["fact_b"][mode]
        c_res = cases["clarification"][mode]
        if a_res["pred_key"] != b_res["pred_key"]:
            pair_changed += 1
            by_lang[lang]["changed"] += 1
        if a_res["correct"] and b_res["correct"]:
            pair_pass += 1
            by_lang[lang]["pair_pass"] += 1
            if c_res["correct"]:
                full3_pass += 1
        if mode == "completion_ll":
            if a_res["ab_correct"] and b_res["ab_correct"]:
                ab_pair_pass += 1
                by_lang[lang]["ab_pair_pass"] += 1

    out = {
        "mode": mode,
        "scenarios": len(by_sc),
        "item_acc": f"{correct_items}/{total_items}",
        "clarification_acc": f"{sum(1 for r in clar_rows if r[mode]['correct'])}/{len(clar_rows)}",
        "adaptation_acc": f"{sum(1 for r in adapt_rows if r[mode]['correct'])}/{len(adapt_rows)}",
        "pair_pass": f"{pair_pass}/{len(by_sc)}",
        "pair_changed_action": f"{pair_changed}/{len(by_sc)}",
        "full_3way_pass": f"{full3_pass}/{len(by_sc)}",
        "by_language": dict(by_lang),
    }
    if mode == "mc_letter":
        out["letter_counts"] = dict(Counter(r[mode]["pred_letter"] for r in records))
    if mode == "completion_ll":
        out["ab_2way_pair_pass"] = f"{ab_pair_pass}/{len(by_sc)}"
    return out


def load_model(ckpt_path: Path, tokenizer_path: Path | None, device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["model_cfg"])
    if tokenizer_path is None:
        cand = ckpt_path.parent / "tokenizer.json"
        if cand.exists():
            tokenizer_path = cand
        elif "train_cfg" in state and (Path(state["train_cfg"]["data_dir"]) / "tokenizer.json").exists():
            tokenizer_path = Path(state["train_cfg"]["data_dir"]) / "tokenizer.json"
        else:
            tokenizer_path = Path("tokenizers/cosmopedia-16k.json")
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size ({tokenizer.vocab_size}) != checkpoint vocab_size ({cfg.vocab_size})"
        )
    model = Transformer(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--dev", type=Path, required=True, help="Path to advice_dev.jsonl")
    ap.add_argument("--include-holdout", action="store_true", help="Also evaluate built-in holdout pairs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-business", action="store_true", help="Omit BUSINESS_INSTRUCTIONS prefix")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSONL path for detailed records")
    args = ap.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)
    business = not args.no_business

    splits = [("dev", load_dev_scenarios(args.dev))]
    if args.include_holdout:
        splits.append(("holdout", HOLDOUT_PAIRS))

    all_records = []
    for split_name, scenarios in splits:
        items = build_items(scenarios, seed=args.seed)
        records = []
        for item in items:
            mc = score_mc_letter(model, tokenizer, item, args.device, business=business)
            cll = score_completion_ll(model, tokenizer, item, args.device, business=business)
            rec = {
                "split": split_name,
                "scenario": item["scenario"],
                "language": item["language"],
                "case": item["case"],
                "gold_key": item["gold_key"],
                "gold_letter": item["gold_letter"],
                "mc_letter": mc,
                "completion_ll": cll,
            }
            records.append(rec)
            all_records.append(rec)

        print(f"=== {args.checkpoint.name} | split={split_name} (seed={args.seed}, business={business}) ===")
        for mode in ("mc_letter", "completion_ll"):
            print(json.dumps(summarize_results(records, mode), ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

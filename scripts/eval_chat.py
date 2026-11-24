"""Run the same offline prompts on each Kuon checkpoint; print answers for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuon.chat import format_prompt
from kuon_agent.llm import KuonEngine


# Keep these separate from SFT and do not tune prompts on the reported scores.
# Review the whole answer: a correct fact plus an invented detail is not a pass.
CASES = [
    ("grounded", "How long does a Neral battery take to charge? Cite the source.",
     "[1] Neral manual\nCharging takes 80 minutes. The battery lasts 9 hours.", "80 minutes [1]"),
    ("grounded", "What color is the Fenwick trail marker? Cite the source.",
     "[1] Park hours\nThe park opens at 6 a.m.\n\n"
     "[2] Trail map\nFenwick uses a blue triangle. Amber uses a green circle.", "blue [2]"),
    ("grounded", "Where should I return the Morrow rental key? Cite the source.",
     "[1] Checkout notice\nLeave the Morrow key in the red box by the gate.\n\n"
     "[2] Arrival notice\nPick up the key at reception.", "red box by the gate [1]"),
    ("grounded", "What temperature should I store Valen pears at? Cite the source.",
     "[1] Kitchen log\nThe prep room is 18°C.\n\n"
     "[2] Storage card\nKeep Valen pears at 4°C.", "4°C [2]"),
    ("abstain", "What is the fee for a lost Morrow rental key?",
     "[1] Checkout notice\nLeave the Morrow key in the red box by the gate.", "fee not specified"),
    ("abstain", "Does Neral offer free battery replacements?",
     "[1] Neral manual\nCharging takes 80 minutes.", "replacement policy not specified"),
    ("abstain", "Who designed the Fenwick trail marker?",
     "[1] Trail map\nFenwick uses a blue triangle.", "designer not specified"),
    ("abstain", "How many Valen pears are in each box?",
     "[1] Storage card\nKeep Valen pears at 4°C.", "quantity not specified"),
    ("python", "Give only the output: `print(sum([4, 8, 2]))`.", None, "14"),
    ("python", "Give only the output: `print('kuon'[1:3])`.", None, "uo"),
    ("python", "Give only the output: `print(len(set([4, 4, 9])))`.", None, "2"),
    ("python", "Give only the output: `print(27 // 5)`.", None, "5"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", help="exact tokenizer used by the checkpoint")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cases", type=Path, help="JSONL evaluation rows (for example, business_eval.jsonl)")
    ap.add_argument("--business", action="store_true", help="use the business chat prompt")
    args = ap.parse_args()
    engine = KuonEngine(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer,
                        temperature=0)

    if args.cases:
        for line in args.cases.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = format_prompt(row["user"], evidence=row.get("evidence"), business=args.business)
            row["answer"] = engine.complete(prompt, max_tokens=96)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    else:
        for kind, question, evidence, expected in CASES:
            answer = engine.complete(format_prompt(question, evidence=evidence), max_tokens=40)
            row = dict(kind=kind, question=question, evidence=evidence, expected=expected, answer=answer)
            if kind == "python":
                row["pass"] = answer == expected
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

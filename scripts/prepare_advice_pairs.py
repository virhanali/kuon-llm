"""Prepare a scenario-disjoint bilingual advice pilot."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "datasets/business"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for line in (ROOT / "advice_pairs.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        lang, name, opening, question, fact_a, answer_a, fact_b, answer_b = line.split("|")
        common = dict(language=lang, scenario=f"{lang}:{name}")
        rows = [dict(common, task="clarification", user=opening, assistant=question)]
        for fact, answer in ((fact_a, answer_a), (fact_b, answer_b)):
            rows.append(dict(common, task="adaptation",
                             user=f"{opening}\nAssistant: {question}\nUser: {fact}", assistant=answer))
        groups[lang].append(rows)

    rng = random.Random(1337)
    train, dev = [], []
    for lang in ("id", "en"):
        cases = groups[lang]
        assert len(cases) == 30
        rng.shuffle(cases)
        for subset, target in ((cases[:24], train), (cases[24:], dev)):
            target.extend(row for case in subset for row in case)

    holdout = [json.loads(line) for line in (ROOT / "advice_eval.jsonl").read_text().splitlines()]
    opening = lambda row: row["user"].split("\n", 1)[0].casefold()
    opens = [{opening(r) for r in rows} for rows in (train, dev, holdout)]
    assert not (opens[0] & opens[1] or opens[0] & opens[2] or opens[1] & opens[2])
    assert len({r["scenario"] for r in train + dev}) == 60
    assert len({r["user"] for r in train + dev}) == 180
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("advice_train", train), ("advice_dev", dev), ("advice_holdout", holdout)):
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(name, len(rows))


if __name__ == "__main__":
    main()

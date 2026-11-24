"""Kuon 0.1 Stage 3A (Clean Diverse SFT + Rehearsal) -> Stage 3B (Context-APO-zero) Runner.

Starts from the clean Stage 2 Gated-Loop 24L checkpoint (`kuon-1-loop-best.pt`, 307.2M tokens)
and runs:
  0. Baseline check on `kuon-1-loop-best.pt` (action_choice + chat)
  1. Stage 3A: Diverse bilingual SFT (~5,500+ rows, 1x business pairs, Aya ID, TyDiQA ID,
     Bitext CS, SalesAgent, SQuAD v2, AmbigQA, FinQA, TAT-QA, SmolTalk, + 15% EN/ID rehearsal)
     with lr=3e-5, warmup, weight_decay=0.01, and best-val checkpoint saving.
  2. Stage 3B: Context-APO-zero preference alignment (counterfactual fact-flip, cross-scenario
     mode-collapse negatives, missing-fact clarification, anti-over-abstention, UltraFeedback)
     with lr=1e-5, beta=2.0, sft_weight=0.1, saving best val_pref_acc checkpoint.
  3. Full 5-suite holdout evaluation on the final aligned checkpoint + lossless inference export.

Works on both Kaggle (auto-locates `kuon-1-loop-best.pt` under `/kaggle/input`) and
Google Colab Pro (`--base /path/to/kuon-1-loop-best.pt`).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def locate_loop_ckpt(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    for base in (Path("/kaggle/input"), Path("/content"), ROOT / "checkpoints"):
        if not base.exists():
            continue
        for name in ("kuon-1-loop-best.pt", "kuon-1-loop-last.pt"):
            for match in base.rglob(name):
                return match
    raise FileNotFoundError("Could not locate kuon-1-loop-best.pt (pass --base or attach Stage 2 notebook output)")


def run_eval_suite(
    ckpt: Path,
    tok_path: Path,
    eval_dir: Path,
    tag: str,
    env: dict[str, str],
    *,
    full: bool = True,
) -> None:
    sub = eval_dir / tag
    sub.mkdir(parents=True, exist_ok=True)
    pairs_dir = eval_dir / "advice_pairs"
    if not (pairs_dir / "advice_dev.jsonl").is_file():
        subprocess.run(
            [sys.executable, "-u", "scripts/prepare_advice_pairs.py", "--out", str(pairs_dir)],
            cwd=str(ROOT), env=env, check=True,
        )

    print(f"\n========== EVAL [{tag}] ({ckpt.name}) ==========", flush=True)
    # 1. eval_action_choice (dev + holdout)
    ac = subprocess.run(
        [
            sys.executable, "-u", "scripts/eval_action_choice.py",
            "--checkpoint", str(ckpt), "--tokenizer", str(tok_path),
            "--dev", str(pairs_dir / "advice_dev.jsonl"),
            "--include-holdout", "--device", "cuda",
            "--out", str(sub / "action_choice.jsonl"),
        ],
        cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
    )
    print(ac.stdout, flush=True)
    (sub / "action_choice_summary.txt").write_text(ac.stdout, encoding="utf-8")

    # 2. eval_chat --business (holdout advice_eval.jsonl)
    ch = subprocess.run(
        [
            sys.executable, "-u", "scripts/eval_chat.py",
            "--checkpoint", str(ckpt), "--tokenizer", str(tok_path),
            "--business", "--cases", "datasets/business/advice_eval.jsonl",
            "--device", "cuda",
        ],
        cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
    )
    print(ch.stdout, flush=True)
    (sub / "chat_business.jsonl").write_text(ch.stdout, encoding="utf-8")

    if not full:
        return

    # 3. eval_confirmed_actions
    ca = subprocess.run(
        [sys.executable, "-u", "scripts/eval_confirmed_actions.py", "--checkpoint", str(ckpt)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
    )
    (sub / "confirmed_actions.jsonl").write_text(ca.stdout, encoding="utf-8")
    if ca.stdout.strip():
        print(f"[{tag} confirmed_actions]", ca.stdout.strip().splitlines()[-1], flush=True)

    # 4. eval_calibration (16 business + 300 AA-Omniscience-Public v1.1)
    subprocess.run(
        [
            sys.executable, "-u", "scripts/eval_calibration.py",
            "--checkpoint", str(ckpt), "--tokenizer", str(tok_path),
            "--device", "cuda", "--aa-public", "--aa-limit", "300", "--max-tokens", "48",
            "--out", str(sub / "calibration_aa_v1_1.jsonl"),
        ],
        cwd=str(ROOT), env=env, check=False,
    )

    # 5. eval_gdpval (25 business tasks)
    subprocess.run(
        [
            sys.executable, "-u", "scripts/eval_gdpval.py",
            "--checkpoint", str(ckpt), "--tokenizer", str(tok_path),
            "--device", "cuda", "--business-only", "--text-only", "--limit", "25", "--max-tokens", "192",
            "--out", str(sub / "gdpval_business.jsonl"),
        ],
        cwd=str(ROOT), env=env, check=False,
    )

    # 6. eval_business_agent (strict leakage-free calibration + agentic tool/deliverable rubric)
    subprocess.run(
        [
            sys.executable, "-u", "scripts/eval_business_agent.py",
            "--checkpoint", str(ckpt), "--tokenizer", str(tok_path),
            "--device", "cuda", "--out", str(sub / "business_agent_eval.json"),
        ],
        cwd=str(ROOT), env=env, check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=None, help="path to clean Stage 2 kuon-1-loop-best.pt")
    ap.add_argument("--out", default=os.environ.get("KUON1_OUT", "/kaggle/working/kuon01-stage3-align"))
    ap.add_argument("--sft-steps", type=int, default=650)
    ap.add_argument("--sft-lr", type=float, default=3e-5)
    ap.add_argument("--align-method", choices=("apo_zero", "orpo"), default="apo_zero")
    ap.add_argument("--align-steps", type=int, default=300)
    ap.add_argument("--align-lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--sft-weight", type=float, default=0.1)
    args = ap.parse_args()

    out_root = Path(args.out)
    sft_data_dir = out_root / "sft-data-v2"
    prefs_data_dir = out_root / "prefs-data"
    ckpt_dir = out_root / "checkpoints"
    eval_dir = out_root / "eval"
    for d in (sft_data_dir, prefs_data_dir, ckpt_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_ckpt = locate_loop_ckpt(args.base)
    tok_path = ROOT / "tokenizers" / "kuon1-bilingual-24k.json"
    if not tok_path.is_file():
        cand = base_ckpt.parent / "tokenizer.json"
        if cand.is_file():
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cand, tok_path)

    env = {**os.environ, "PYTHONPATH": str(ROOT), "MKL_THREADING_LAYER": "GNU"}
    print(f"[stage3-align] Clean Stage 2 base: {base_ckpt}", flush=True)
    print(f"[stage3-align] Tokenizer         : {tok_path}", flush=True)

    # 0. Baseline evaluation on clean Stage 2 base (before SFT)
    run_eval_suite(base_ckpt, tok_path, eval_dir, "0_stage2_base", env, full=False)

    # 1. Stage 3A — Clean Diverse Bilingual SFT + 15% Experience Rehearsal
    print("\n[stage3-align] Phase 3A — Building clean diverse bilingual SFT + rehearsal corpus...", flush=True)
    subprocess.run(
        [sys.executable, "-u", "scripts/prepare_kuon1_sft.py", "--tokenizer", str(tok_path), "--out", str(sft_data_dir)],
        cwd=str(ROOT), env=env, check=True,
    )
    sft_ckpt = ckpt_dir / "kuon-0.1-sft-v2.pt"
    print(f"\n[stage3-align] Phase 3A — Running SFT ({args.sft_steps} steps @ bs=4x4, lr={args.sft_lr})...", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", "scripts/sft.py",
            "--base", str(base_ckpt), "--tokenizer", str(tok_path),
            "--data", str(sft_data_dir), "--out", str(sft_ckpt),
            "--steps", str(args.sft_steps), "--batch-size", "4", "--grad-accum", "4",
            "--lr", str(args.sft_lr), "--min-lr", str(args.sft_lr / 10.0),
            "--warmup-steps", "40", "--weight-decay", "0.01",
            "--eval-interval", "65", "--patience", "4",
            "--device", "cuda", "--sample-prompt", "Penjualan toko minggu ini turun 15%.",
        ],
        cwd=str(ROOT), env=env, check=True,
    )
    run_eval_suite(sft_ckpt, tok_path, eval_dir, "1_after_sft_v2", env, full=False)

    # 2. Stage 3B — Context-APO-zero / ORPO Preference Alignment
    print("\n[stage3-align] Phase 3B — Building counterfactual + calibration preference pairs...", flush=True)
    subprocess.run(
        [sys.executable, "-u", "scripts/prepare_kuon1_prefs.py", "--tokenizer", str(tok_path), "--out", str(prefs_data_dir)],
        cwd=str(ROOT), env=env, check=True,
    )
    aligned_ckpt = ckpt_dir / f"kuon-0.1-{args.align_method}.pt"
    print(f"\n[stage3-align] Phase 3B — Running {args.align_method} ({args.align_steps} steps @ bs=4x4, lr={args.align_lr})...", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", "scripts/align.py",
            "--base", str(sft_ckpt), "--tokenizer", str(tok_path),
            "--data", str(prefs_data_dir), "--out", str(aligned_ckpt),
            "--method", args.align_method, "--steps", str(args.align_steps),
            "--batch-size", "4", "--grad-accum", "4",
            "--lr", str(args.align_lr), "--min-lr", str(args.align_lr / 10.0),
            "--warmup-steps", "25", "--weight-decay", "0.01",
            "--beta", str(args.beta), "--sft-weight", str(args.sft_weight),
            "--eval-interval", "50", "--device", "cuda",
        ],
        cwd=str(ROOT), env=env, check=True,
    )

    # 3. Full 5-Suite Holdout Evaluation on final aligned checkpoint
    run_eval_suite(aligned_ckpt, tok_path, eval_dir, f"2_after_{args.align_method}", env, full=True)

    # 4. Lossless inference export (weights unchanged FP32, optimizer stripped)
    inf_ckpt = ckpt_dir / "kuon-0.1-inference.pt"
    if not inf_ckpt.exists():
        subprocess.run(
            [sys.executable, "-u", "scripts/export_inference.py", "--checkpoint", str(aligned_ckpt), "--out", str(inf_ckpt)],
            cwd=str(ROOT), env=env, check=False,
        )

    print(f"\n[stage3-align] ALL PHASES COMPLETE! Outputs saved under {out_root}", flush=True)


if __name__ == "__main__":
    main()

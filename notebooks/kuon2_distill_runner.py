"""End-to-end Colab L4 runner: Gemma 4 E4B-IT distillation -> Stage 3 -> 6-Suite Eval -> INT8 Release."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def run(cmd: list[str], *, env=None, cwd=None, log_path: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(
        cmd, env=env, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lf = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if lf:
                lf.write(line)
                lf.flush()
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if lf:
            lf.close()


def setup_kaggle() -> None:
    from google.colab import userdata

    tok = (userdata.get("KAGGLE_API_TOKEN") or "").strip()
    assert tok, "KAGGLE_API_TOKEN missing"
    os.environ["KAGGLE_API_TOKEN"] = tok
    kd = Path("/root/.kaggle")
    kd.mkdir(parents=True, exist_ok=True)
    p = kd / "access_token"
    p.write_text(tok + "\n", encoding="utf-8")
    p.chmod(0o600)
    run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers", "accelerate", "bitsandbytes", "kaggle", "tokenizers", "datasets",
    ])


def dl(slug: str, dest: Path, files: list[str] | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if files:
        for fn in files:
            t = dest / fn
            if t.is_file() and t.stat().st_size > 0:
                print(f"[kaggle] cached {fn}", flush=True)
                continue
            r = subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-f", fn, "-p", str(dest), "--force"],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, f"{slug}/{fn}: {r.stderr}"
            zp = dest / (fn + ".zip")
            if zp.is_file():
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(dest)
                zp.unlink()
    else:
        run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip", "--force"])


def main() -> None:
    import torch

    assert torch.cuda.is_available(), "No GPU detected"
    print(
        f"[kuon2-distill] GPU={torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)",
        flush=True,
    )
    setup_kaggle()

    repo = Path("/content/kuon-repo")
    assert (repo / "scripts/distill_gemma4.py").is_file()

    s2_dir = Path("/content/kuon2-s2-ckpt")
    dl("virhanali/kuon2-300m-stage2-ckpts", s2_dir, ["kuon-2-loop-best.pt", "tokenizer.json"])
    base = s2_dir / "kuon-2-loop-best.pt"

    s3_v3 = Path("/content/kuon2-stage3-data-v3")
    if s3_v3.exists():
        shutil.rmtree(s3_v3)
    dl(
        "virhanali/kuon2-stage3-data-v3", s3_v3,
        ["sft_train.jsonl", "sft_val.jsonl", "prefs_train.jsonl", "prefs_val.jsonl"],
    )

    dist_dir = Path("/content/kuon2-stage3-distill")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    env = {**os.environ, "PYTHONPATH": str(repo), "MKL_THREADING_LAYER": "GNU"}

    # Step 1: Distill with Gemma 4 E4B-IT (try BF16 bs=8, fallback to 4-bit bs=16 if OOM)
    base_cmd = [
        sys.executable, "-u", str(repo / "scripts/distill_gemma4.py"),
        "--model", "google/gemma-4-E4B-it",
        "--out", str(dist_dir),
        "--scenarios", "600", "--evidence-cases", "1000", "--consult-cases", "800",
        "--merge-base-dir", str(s3_v3),
    ]
    try:
        run(base_cmd + ["--batch-size", "8"], cwd=str(repo), env=env)
    except subprocess.CalledProcessError:
        print("[kuon2-distill] Retrying with --load-in-4bit --batch-size 16...", flush=True)
        run(base_cmd + ["--load-in-4bit", "--batch-size", "16"], cwd=str(repo), env=env)

    # Backup distilled dataset to private Kaggle
    (dist_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "Kuon-2 Stage 3 Gemma4 Distilled Data",
        "id": "virhanali/kuon2-stage3-distill-v1",
        "licenses": [{"name": "other"}],
    }))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(dist_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        run(["kaggle", "datasets", "version", "-p", str(dist_dir), "-m", "Gemma 4 E4B-IT distilled bilingual dataset"])
    print("[kuon2-distill] Backed up dataset virhanali/kuon2-stage3-distill-v1", flush=True)

    # Step 2: Train Stage 3 on distilled + merged dataset
    s3_out = Path("/content/kuon2-stage3-distill-out")
    if s3_out.exists():
        shutil.rmtree(s3_out)
    s3_out.mkdir(parents=True, exist_ok=True)
    s3_log = s3_out / "stage3_run.log"
    run([
        sys.executable, "-u", str(repo / "notebooks/kuon1_stage3_align_runner.py"),
        "--base", str(base), "--out", str(s3_out),
        "--sft-data", str(dist_dir), "--prefs-data", str(dist_dir),
        "--model-tag", "kuon-2",
        "--sft-steps", "2000", "--sft-lr", "2e-5",
        "--align-method", "apo_zero", "--align-steps", "900", "--align-lr", "6e-6",
        "--beta", "2.0", "--sft-weight", "0.1", "--grad-checkpoint",
    ], cwd=str(repo), env=env, log_path=s3_log)

    # Step 3: Upload distilled release to Kaggle
    rel = Path("/content/kaggle-release-distill")
    if rel.exists():
        shutil.rmtree(rel)
    rel.mkdir(parents=True)
    ck3 = s3_out / "checkpoints"
    for fn in ("kuon-2-int8.pt", "kuon-2-inference.pt", "tokenizer.json"):
        if (ck3 / fn).is_file():
            shutil.copy2(ck3 / fn, rel / fn)
    if not (rel / "tokenizer.json").is_file():
        shutil.copy2(s2_dir / "tokenizer.json", rel / "tokenizer.json")
    if s3_log.is_file():
        shutil.copy2(s3_log, rel / "stage3_run.log")
    ev = s3_out / "eval"
    for src_rel, dst_name in (
        ("2_after_apo_zero/business_agent_eval.json", "business_agent_eval.json"),
        ("2_after_apo_zero/confirmed_actions.jsonl", "confirmed_actions.jsonl"),
        ("2_after_apo_zero/action_choice_summary.txt", "action_choice_summary.txt"),
        ("2_after_apo_zero/calibration_aa_v1_1.jsonl", "calibration_aa_v1_1.jsonl"),
        ("2_after_apo_zero/gdpval_business.jsonl", "gdpval_business.jsonl"),
        ("2_after_apo_zero/chat_business.jsonl", "chat_business.jsonl"),
        ("3_int8_parity/business_agent_eval.json", "int8_business_agent_eval.json"),
        ("3_int8_parity/confirmed_actions.jsonl", "int8_confirmed_actions.jsonl"),
    ):
        if (ev / src_rel).is_file():
            shutil.copy2(ev / src_rel, rel / dst_name)
    with tarfile.open(rel / "eval_all.tar.gz", "w:gz") as tf:
        tf.add(ev, arcname="eval")
    (rel / "dataset-metadata.json").write_text(json.dumps({
        "title": "Kuon-2 300M Release Artifacts",
        "id": "virhanali/kuon2-300m-release",
        "licenses": [{"name": "other"}],
    }))
    run([
        "kaggle", "datasets", "version", "-p", str(rel),
        "-m", "Kuon-2 300M Stage 3 Distilled (Gemma 4 E4B-IT + 2000 SFT + 900 APO-zero + INT8)",
        "-r", "tar",
    ])
    print("\n[kuon2-distill] ALL DONE!", flush=True)


if __name__ == "__main__":
    main()

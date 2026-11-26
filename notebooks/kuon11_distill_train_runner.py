"""End-to-end Colab L4 runner for Kuon 1.1:
Gemma 4 E4B-IT non-coding distillation -> v4.2 anchor merge -> Stage 3 SFT + APO-zero -> 6-suite eval -> INT8 export.

Saves artifacts privately to:
  - Dataset: virhanali/kuon11-noncoding-distill-data
  - Release: virhanali/kuon2-300m-release (new version)
"""

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
    assert tok, "KAGGLE_API_TOKEN missing in Colab Secrets"
    os.environ["KAGGLE_API_TOKEN"] = tok
    kd = Path("/root/.kaggle")
    kd.mkdir(parents=True, exist_ok=True)
    p = kd / "access_token"
    p.write_text(tok + "\n", encoding="utf-8")
    p.chmod(0o600)
    run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers", "accelerate", "bitsandbytes", "kaggle", "tokenizers", "datasets", "six", "regex",
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
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[kuon11] GPU={torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)", flush=True)
    setup_kaggle()

    repo = Path("/content/kuon-repo")
    if repo.exists():
        shutil.rmtree(repo)
    dl("virhanali/kuon-code-latest", repo)
    for tf_path in list(repo.glob("*.tar")):
        with tarfile.open(tf_path) as tf:
            tf.extractall(repo)
        tf_path.unlink()
    assert (repo / "scripts/distill_kuon11.py").is_file()

    s2_dir = Path("/content/kuon2-s2-ckpt")
    dl("virhanali/kuon2-300m-stage2-ckpts", s2_dir, ["kuon-2-loop-best.pt", "tokenizer.json"])
    base = s2_dir / "kuon-2-loop-best.pt"
    tok_path = s2_dir / "tokenizer.json"

    v42_dir = Path("/content/kuon2-stage3-v42-data")
    if v42_dir.exists():
        shutil.rmtree(v42_dir)
    dl(
        "virhanali/kuon2-stage3-v42-data", v42_dir,
        ["sft_train.jsonl", "sft_val.jsonl", "prefs_train.jsonl", "prefs_val.jsonl"],
    )

    dist_dir = Path("/content/kuon11-distill-data")
    dist_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(repo), "MKL_THREADING_LAYER": "GNU"}

    # Step 1: Non-coding Gemma 4 E4B-IT distillation (resumable via raw_cache.jsonl)
    base_cmd = [
        sys.executable, "-u", str(repo / "scripts/distill_kuon11.py"),
        "--out", str(dist_dir),
        "--backend", "hf",
        "--model", "google/gemma-4-E4B-it",
        "--casual", "250",
        "--open-consult", "700",
        "--doc-summary", "550",
        "--writing", "650",
        "--explain", "550",
        "--multiturn", "400",
        "--merge-base-dir", str(v42_dir),
        "--tokenizer", str(tok_path),
        "--block-size", "1024",
    ]
    try:
        run(base_cmd + ["--load-in-4bit", "--batch-size", "16"], cwd=str(repo), env=env)
    except subprocess.CalledProcessError:
        print("[kuon11] Retrying distillation with --load-in-4bit --batch-size 8...", flush=True)
        run(base_cmd + ["--load-in-4bit", "--batch-size", "8"], cwd=str(repo), env=env)

    # Free teacher VRAM before Stage 3 training
    torch.cuda.empty_cache()

    # Backup distilled dataset privately to Kaggle (excluding raw_cache.jsonl if huge, keep clean splits)
    ds_pkg = Path("/content/kuon11-kaggle-ds")
    if ds_pkg.exists():
        shutil.rmtree(ds_pkg)
    ds_pkg.mkdir(parents=True)
    for fn in ("sft_train.jsonl", "sft_val.jsonl", "prefs_train.jsonl", "prefs_val.jsonl"):
        shutil.copy2(dist_dir / fn, ds_pkg / fn)
    (ds_pkg / "dataset-metadata.json").write_text(json.dumps({
        "title": "Kuon 1.1 Non-Coding Gemma4 Distilled Data",
        "id": "virhanali/kuon11-noncoding-distill-data",
        "licenses": [{"name": "other"}],
    }))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(ds_pkg)], capture_output=True, text=True)
    if r.returncode != 0:
        run(["kaggle", "datasets", "version", "-p", str(ds_pkg), "-m", "Kuon 1.1 non-coding Gemma4-E4B + v4.2 anchor"])
    print("[kuon11] Backed up private dataset virhanali/kuon11-noncoding-distill-data", flush=True)

    # Step 2: Stage 3 SFT + APO-zero (proven v4.2 beta=1.0, sft_weight=0.35 recipe)
    s3_out = Path("/content/kuon11-stage3-out")
    if s3_out.exists():
        shutil.rmtree(s3_out)
    s3_out.mkdir(parents=True, exist_ok=True)
    s3_log = s3_out / "stage3_run.log"
    run([
        sys.executable, "-u", str(repo / "notebooks/kuon1_stage3_align_runner.py"),
        "--base", str(base), "--out", str(s3_out),
        "--sft-data", str(dist_dir), "--prefs-data", str(dist_dir),
        "--model-tag", "kuon-1-1",
        "--sft-steps", "2400", "--sft-lr", "2e-5",
        "--align-method", "apo_zero", "--align-steps", "700", "--align-lr", "3e-6",
        "--beta", "1.0", "--sft-weight", "0.35", "--grad-checkpoint",
    ], cwd=str(repo), env=env, log_path=s3_log)

    # Step 3: Upload Kuon 1.1 release artifacts to Private Kaggle
    rel = Path("/content/kaggle-release-kuon11")
    if rel.exists():
        shutil.rmtree(rel)
    rel.mkdir(parents=True)
    ck3 = s3_out / "checkpoints"
    for fn in ("kuon-1-1-int8.pt", "kuon-1-1-inference.pt", "tokenizer.json"):
        if (ck3 / fn).is_file():
            shutil.copy2(ck3 / fn, rel / fn)
    if not (rel / "tokenizer.json").is_file():
        shutil.copy2(tok_path, rel / "tokenizer.json")
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
        "-m", "Kuon 1.1 Non-Coding Gemma4-E4B Distilled + v4.2 Anchor (2400 SFT + 700 APO-zero + INT8)",
        "-r", "tar",
    ])
    print("\n[kuon11] ALL DONE!", flush=True)


if __name__ == "__main__":
    main()

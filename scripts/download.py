"""Download official eval suites into data/eval/ (read-only) and train seeds
into data/raw/. Never mix: eval rows must not enter any training path.

Usage: .venv/bin/python scripts/download.py [--eval-only]
Requires: uv pip install datasets  (HF_TOKEN in .env for gated sets like xlam-60k)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
RAW = ROOT / "data" / "raw"


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
    return None


def _save_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        sys.exit("pip install datasets first")

    token = _token()

    # ---- Layer C: sealed eval ----
    jobs_eval = [
        # (repo, config, split, filter, out)
        ("google/mobile-actions", None, "train", lambda r: r.get("metadata") == "eval", "mobile_actions_eval.jsonl"),
        # DroidCall 200-row test set is not on HF (train only); fetch from their GitHub in the harness phase
    ]
    for repo, cfg, split, flt, out in jobs_eval:
        try:
            ds = load_dataset(repo, cfg, split=split, token=token)
            rows = (r for r in ds if flt is None or flt(r))
            n = _save_jsonl(rows, EVAL / out)
            print(f"eval  {repo}:{split} -> {out} ({n})")
        except Exception as e:  # gated/renamed sets shouldn't kill the rest
            print(f"SKIP eval {repo}: {e}")

    if args.eval_only:
        return

    # ---- Layer A: train seeds (schemas + queries; never SFT'd raw) ----
    jobs_raw = [
        ("google/mobile-actions", None, "train", lambda r: r.get("metadata") != "eval", "mobile_actions_train.jsonl"),
        ("mllmTeam/DroidCall", None, "train", None, "droidcall_train.jsonl"),
        ("Salesforce/xlam-function-calling-60k", None, "train", None, "xlam60k.jsonl"),
        ("Team-ACE/ToolACE", None, "train", None, "toolace.jsonl"),
        ("NousResearch/hermes-function-calling-v1", "func_calling", "train", None, "hermes_fc.jsonl"),
        ("Salesforce/APIGen-MT-5k", None, "train", None, "apigen_mt.jsonl"),
    ]
    for repo, cfg, split, flt, out in jobs_raw:
        try:
            ds = load_dataset(repo, cfg, split=split, token=token)
            rows = (r for r in ds if flt is None or flt(r))
            n = _save_jsonl(rows, RAW / out)
            print(f"raw   {repo}:{split} -> {out} ({n})")
        except Exception as e:
            print(f"SKIP raw {repo}: {e}")


if __name__ == "__main__":
    main()

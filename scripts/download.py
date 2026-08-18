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

    _seal_tools()
    if not args.eval_only:
        _layer_a(load_dataset, token)


def _seal_tools() -> None:
    """Seal-Tools lives on GitHub, not HF: clone shallow, copy the eval jsons."""
    import shutil
    import subprocess
    import tempfile

    if (EVAL / "seal_tools_out_domain.json").exists():
        return
    with tempfile.TemporaryDirectory() as td:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/fairyshine/Seal-Tools", td],
                check=True, capture_output=True, timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"SKIP seal-tools: {e}")
            return
        found = 0
        for p in Path(td).rglob("*.json"):
            low = p.name.lower()
            if "test_in_domain" in low or "test_out_domain" in low:
                dest = EVAL / f"seal_tools_{'in' if 'in_domain' in low else 'out'}_domain.json"
                shutil.copy(p, dest)
                found += 1
            elif "train" in low and "seal" not in low and p.stat().st_size > 1_000_000:
                RAW.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, RAW / f"seal_tools_{p.name}")
        print(f"eval  Seal-Tools: {found} test files copied" if found else "SKIP seal-tools: test files not found in repo layout")


def _layer_a(load_dataset, token) -> None:
    # ---- Layer A: train seeds (schemas + queries; never SFT'd raw) ----
    jobs_raw = [
        ("google/mobile-actions", None, "train", lambda r: r.get("metadata") != "eval", "mobile_actions_train.jsonl"),
        ("mllmTeam/DroidCall", None, "train", None, "droidcall_train.jsonl"),
        ("Salesforce/xlam-function-calling-60k", None, "train", None, "xlam60k.jsonl"),
        ("Team-ACE/ToolACE", None, "train", None, "toolace.jsonl"),
        ("NousResearch/hermes-function-calling-v1", "func_calling", "train", None, "hermes_fc.jsonl"),
        ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", None, "hermes_single.jsonl"),
        ("Salesforce/APIGen-MT-5k", None, "train", None, "apigen_mt.jsonl"),
        # single-pass catalogs we have no coverage of; each is a different
        # schema dialect, which is the point — naming monoculture is what sank
        # Seal-Tools the first time round
        ("glaiveai/glaive-function-calling-v2", None, "train", None, "glaive_v2.jsonl"),
        ("Nexusflow/NexusRaven_API_evaluation", None, "train", None, "nexus_raven.jsonl"),
        ("driaforall/pythonic-function-calling", None, "train", None, "dria_pythonic.jsonl"),
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

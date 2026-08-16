"""Post-SFT eval on the pod: held-out local eval + OOD + official Mobile Actions.

Prints Needle-comparable tables for S2 / heads-on / heads-off.
Usage (on pod): python scripts/pod_eval.py [--ckpt sft] [--n-eval 1000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tiny_toolcall.eval import make_model_predictor, predict_s2, score_predictor
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.official import mobile_actions_rows
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
FMT_KEYS = ["accuracy", "name_acc", "well_formed", "non_empty", "one_call", "two_plus", "refuse"]


def fmt(s: dict) -> str:
    return "  ".join(f"{k}={s.get(k, 0):.3f}" for k in FMT_KEYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sft")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-ood", type=int, default=750)
    ap.add_argument("--n-ma", type=int, default=0)
    args = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    suites: list[tuple[str, list[dict]]] = []
    for name, fname, cap in (("local-eval", "local_eval.jsonl", args.n_eval), ("local-ood", "local_ood.jsonl", args.n_ood)):
        p = ROOT / "data" / "seeds" / fname
        if p.exists():
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            suites.append((name, rows[:cap] if cap else rows))
    ma = ROOT / "data" / "eval" / "mobile_actions_eval.jsonl"
    if ma.exists():
        rows = mobile_actions_rows(ma)
        suites.append(("mobile-actions", rows[: args.n_ma] if args.n_ma else rows))

    for name, rows in suites:
        print(f"\n=== {name} (n={len(rows)}) ===")
        print("S2       :", fmt(score_predictor(rows, predict_s2)))
        for heads in (True, False):
            t0 = time.time()
            pred = make_model_predictor(model, tok, device, use_name_head=heads)
            s = score_predictor(rows, pred)
            label = "heads-on " if heads else "heads-off"
            print(f"{label}:", fmt(s), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

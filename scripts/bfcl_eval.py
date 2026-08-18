"""Score a checkpoint on BFCL v4 single-turn: per-category and unweighted mean.

This is the one suite nothing in this project was tuned against — no BFCL data
is in the training mix, and no decode constant was ever chosen by watching a
BFCL number. It is therefore the only honest generalization test we have, and
it gets reported whatever it says.

Caveat kept with the number: xLAM is 17% of our training mix and was built by
Salesforce partly to score well on BFCL, so "never seen" is true of the rows,
not of the distribution.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tiny_toolcall.bfcl import SINGLE_TURN, bfcl_rows, score_row, unweighted_mean
from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sft_v3_scratch")
    ap.add_argument("--data", required=True, help="BFCL_v4 data dir")
    ap.add_argument("--limit", type=int, default=0, help="per-category cap, 0 = all")
    ap.add_argument("--max-prompt", type=int, default=600)
    ap.add_argument("--only", default="", help="comma-separated categories, for reruns")
    a = ap.parse_args()
    cats = [c for c in a.only.split(",") if c] or SINGLE_TURN

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    data = Path(a.data)
    per_cat: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for cat in cats:
        rows = bfcl_rows(data, cat)
        if a.limit:
            rows = rows[: a.limit]
        ok = over = 0
        t0 = time.time()
        for r in rows:
            ex = normalize_example(r)
            pr = prompt_text(ex["query"], ex["tools"])
            pid = tok.encode(pr)
            if len(pid) > a.max_prompt:
                # counted as a miss, not skipped: the row is part of the suite and
                # a 640-token model genuinely cannot answer it
                over += 1
                continue
            s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
            spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
            pred = canon_calls(constrained_decode(
                model, tok, pr, ex["query"], ex["tools"], dev, name_spans=spans, gated=True))
            ok += score_row(pred, r)
        acc = 100.0 * ok / max(1, len(rows))
        per_cat[cat] = acc
        detail[cat] = {"n": len(rows), "correct": ok, "over_context": over,
                       "secs": round(time.time() - t0)}
        print(f"{cat:26s} n={len(rows):5d}  acc={acc:5.1f}%  "
              f"over-context={over:4d}  ({detail[cat]['secs']}s)", flush=True)

    mean = unweighted_mean(per_cat)
    print(f"\nBFCL v4 single-turn, unweighted mean over {len(per_cat)} categories: {mean:.1f}")
    out = ROOT / f"bfcl_{a.ckpt}{'_' + a.only.replace(',','-') if a.only else ''}.json"
    out.write_text(json.dumps({"ckpt": a.ckpt, "per_category": per_cat,
                               "detail": detail, "unweighted_mean": mean}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

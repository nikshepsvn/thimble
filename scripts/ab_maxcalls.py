"""A/B the MAX_CALLS cap on a suite it helps AND a suite it cannot help.

Seal-Tools has rows needing 5-6 calls that a cap of 4 makes unwinnable. Mobile
Actions has none — it is 1-2 calls throughout — so raising the cap can only
hurt it, via rows where the model would have been forced to stop and now keeps
going. Reporting both is the whole point: the repetition blocker looked free on
the suite it was designed for and cost 51 points on the one it was not.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.official import mobile_actions_rows, seal_tools_rows
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls, score_rows
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sft_v3_scratch")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--caps", default="4,6")
    a = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    ev = ROOT / "data" / "eval"
    suites = {
        "seal-in": seal_tools_rows(ev / "seal_tools_in_domain.json"),
        "mobile-actions": mobile_actions_rows(ev / "mobile_actions_eval.jsonl"),
    }
    caps = [int(c) for c in a.caps.split(",")]

    for name, src in suites.items():
        rows = [normalize_example(r) for r in src]
        random.Random(7).shuffle(rows)  # Seal ships difficulty-sorted
        rows = rows[: a.n]
        for cap in caps:
            out = []
            for ex in rows:
                pr = prompt_text(ex["query"], ex["tools"])
                pid = tok.encode(pr)
                s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
                spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
                pred = canon_calls(constrained_decode(
                    model, tok, pr, ex["query"], ex["tools"], dev,
                    name_spans=spans, gated=True, max_calls=cap))
                out.append({"gold": ex["answers"], "pred": pred})
            s = score_rows(out)
            print(f"{name:16s} max_calls={cap}  acc={s['accuracy']:.3f}  "
                  f"name={s.get('name_acc', 0):.3f}  well_formed={s.get('well_formed', 0):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()

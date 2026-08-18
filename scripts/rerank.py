"""Field-set reranking (PGR-style, arXiv 2608.03071): harvest the key-set
variance the field-oracle measurement proved is there.

Measured gate (v4, seal-in, n=120, k=8, temp 0.8):
    greedy exact 27.5 / selection ceiling 33.3 / COMPOSITION ceiling 39.2
    right key set present in pool: 79.1% of calls vs 45.1% greedy-correct
so the correct key set usually exists in the pool even when greedy adds a
spurious optional — which is 65% of our key-set errors.

Strategy = their conservative field-set variant:
  name      greedy's (name selection is not the bottleneck; do not touch it)
  key set   modal key set among same-position, same-name pool candidates;
            ties break to greedy
  values    greedy's value for every kept key; if the chosen set contains a key
            greedy lacks, the modal value among candidates that have it

This composes rather than selects, so it is NOT bounded by the +5.0 pass@k
selection ceiling. Standing rule applies: ships only if the Mobile Actions arm
is non-negative — the repetition blocker cost 51 points for skipping that test.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch

from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.official import mobile_actions_rows, seal_tools_rows
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def rerank(greedy: list[dict], pool: list[list[dict]]) -> list[dict]:
    out = []
    for i, g in enumerate(greedy):
        cands = [c[i] for c in pool if len(c) > i and c[i]["name"] == g["name"]]
        if not cands:
            out.append(g)
            continue
        sets = collections.Counter(frozenset(c["arguments"]) for c in cands)
        top, top_n = sets.most_common(1)[0]
        gset = frozenset(g["arguments"])
        # conservative: greedy keeps its keys unless a strict majority disagrees
        chosen = top if (top != gset and top_n > len(cands) / 2) else gset
        args = {}
        for k in chosen:
            if k in g["arguments"]:
                args[k] = g["arguments"][k]
            else:
                vals = collections.Counter(
                    json.dumps(c["arguments"][k], sort_keys=True)
                    for c in cands if k in c["arguments"])
                if vals:
                    args[k] = json.loads(vals.most_common(1)[0][0])
        out.append({"name": g["name"], "arguments": args})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4")
    ap.add_argument("--suite", default="seal-in", choices=["seal-in", "seal-out", "mobile-actions"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    ev = ROOT / "data" / "eval"
    src = {"seal-in": lambda: seal_tools_rows(ev / "seal_tools_in_domain.json"),
           "seal-out": lambda: seal_tools_rows(ev / "seal_tools_out_domain.json"),
           "mobile-actions": lambda: mobile_actions_rows(ev / "mobile_actions_eval.jsonl")}[a.suite]()
    rows = [normalize_example(r) for r in src]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

    g_ok = r_ok = fixed = broke = 0
    t0 = time.time()
    for i, ex in enumerate(rows):
        pr = prompt_text(ex["query"], ex["tools"])
        pid = tok.encode(pr)
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
        spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
        gold = canon_calls(ex["answers"])

        def dec(temp):
            return canon_calls(constrained_decode(
                model, tok, pr, ex["query"], ex["tools"], dev,
                name_spans=spans, gated=True, temp=temp))

        greedy = dec(0.0)
        pool = [dec(a.temp) for _ in range(a.k)]
        rr = canon_calls(rerank(greedy, [greedy] + pool))
        g, r = greedy == gold, rr == gold
        g_ok += g; r_ok += r
        fixed += (not g) and r
        broke += g and (not r)
        if (i + 1) % 20 == 0:
            n = i + 1
            print(f"{n}/{len(rows)} greedy={100*g_ok/n:.1f} reranked={100*r_ok/n:.1f} "
                  f"fixed={fixed} broke={broke} ({time.time()-t0:.0f}s)", flush=True)

    n = len(rows)
    print(f"\n=== {a.suite}  ckpt={a.ckpt}  n={n}  k={a.k}  temp={a.temp}")
    print(f"greedy    {100*g_ok/n:5.1f}%")
    print(f"reranked  {100*r_ok/n:5.1f}%   (fixed {fixed}, broke {broke})")
    print(f"delta     {100*(r_ok-g_ok)/n:+.1f} points")


if __name__ == "__main__":
    main()

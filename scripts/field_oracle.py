"""Can a correct call be ASSEMBLED from a pool of samples, even when no single
sample is correct?

pass@k answered a different question — "is any whole sample exactly right?" — and
came back +5.0, which killed selection-from-a-pool (beam, best-of-N, RL). But
Probe-Guided Reranking (arXiv 2608.03071) does not select a candidate, it builds
one, and reports that the strategy which "fits when the greedy values are right
but the call adds fields it should not" is a distinct win. That is our error:
89 of 137 key-set failures are exactly one spurious field with the values right.

So this measures the oracles pass@k never touched, per gold call:

  name          the right tool name appears in some sample
  key-set       the right key set appears in some sample
  each-field    every field's correct value appears in some sample (possibly
                different samples for different fields)
  assembled     key-set AND each-field both available -> a correct call is
                constructible from the pool

`assembled` is the ceiling for any reranker that composes rather than picks. If
it sits far above pass@k, reranking is worth building. If it tracks pass@k, the
variance is not there and this dies with the rest.

These are oracles: they use the gold answer to check availability. Not scores.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.official import seal_tools_rows
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
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

    rows = [normalize_example(r) for r in seal_tools_rows(ROOT / "data/eval/seal_tools_in_domain.json")]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

    st = {k: 0 for k in ("rows", "greedy_row", "pool_row", "assembled_row",
                         "calls", "greedy_call", "name_avail", "keyset_avail",
                         "fields_avail", "assembled_call")}
    t0 = time.time()
    for ri, ex in enumerate(rows):
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
        pool = [greedy] + [dec(a.temp) for _ in range(a.k)]

        st["rows"] += 1
        st["greedy_row"] += greedy == gold
        st["pool_row"] += any(c == gold for c in pool)

        # per gold call position, what is available anywhere in the pool?
        row_ok = True
        for i, g in enumerate(gold):
            st["calls"] += 1
            same_pos = [c[i] for c in pool if len(c) > i]
            st["greedy_call"] += len(greedy) > i and greedy[i] == g
            name_ok = any(c["name"] == g["name"] for c in same_pos)
            # only candidates that got the name right can supply fields
            matched = [c for c in same_pos if c["name"] == g["name"]]
            keyset_ok = any(set(c["arguments"]) == set(g["arguments"]) for c in matched)
            fields_ok = all(
                any(c["arguments"].get(k, object()) == v for c in matched)
                for k, v in g["arguments"].items()
            ) if matched else not g["arguments"]
            st["name_avail"] += name_ok
            st["keyset_avail"] += keyset_ok
            st["fields_avail"] += fields_ok
            call_ok = name_ok and keyset_ok and fields_ok
            st["assembled_call"] += call_ok
            row_ok = row_ok and call_ok
        st["assembled_row"] += row_ok and len(gold) > 0

        if (ri + 1) % 20 == 0:
            n, c = st["rows"], max(1, st["calls"])
            print(f"{n}/{len(rows)} greedy={100*st['greedy_row']/n:.1f} "
                  f"pool={100*st['pool_row']/n:.1f} assembled={100*st['assembled_row']/n:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    n, c = st["rows"], max(1, st["calls"])
    print(f"\n=== seal-in  ckpt={a.ckpt}  n={n}  k={a.k}  temp={a.temp}")
    print("ROW level")
    print(f"  greedy exact           {100*st['greedy_row']/n:5.1f}%   (the real score)")
    print(f"  pass@{a.k+1} (pick one)      {100*st['pool_row']/n:5.1f}%   <- ceiling for SELECTION")
    print(f"  ASSEMBLED from pool    {100*st['assembled_row']/n:5.1f}%   <- ceiling for COMPOSITION")
    print("\nCALL level (per gold call, availability anywhere in the pool)")
    print(f"  greedy correct         {100*st['greedy_call']/c:5.1f}%")
    print(f"  right name available   {100*st['name_avail']/c:5.1f}%")
    print(f"  right KEY SET avail    {100*st['keyset_avail']/c:5.1f}%")
    print(f"  every field avail      {100*st['fields_avail']/c:5.1f}%")
    print(f"  fully assemblable      {100*st['assembled_call']/c:5.1f}%")
    gain = (st["assembled_row"] - st["pool_row"]) / n
    print(f"\ncomposition beyond selection: {100*gain:+.1f} points")
    print("VERDICT:", "RERANKING IS WORTH BUILDING" if gain > 0.05 else
          "no compositional headroom either — the variance is not there")
    (ROOT / f"field_oracle_{a.ckpt}.json").write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()

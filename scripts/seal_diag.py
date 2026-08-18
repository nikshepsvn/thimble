"""Attribute Seal-Tools row failures to a specific stage.

Every reported Seal number is a single scalar (19.7) that could be produced by
very different underlying failures — wrong call count, right calls in the wrong
order, right calls with wrong values. Optimising against the scalar is guesswork;
this splits it.

Three passes over the same rows:
  free    — the real decoder, exactly as scored
  oracle  — call sequence pinned to gold names, so only argument filling can fail
and from `free` alone we derive count/multiset/sequence agreement.

The oracle pass is diagnostic only. Its accuracy is the ceiling that perfect
tool selection would buy us, and its complement is what argument filling costs.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
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


def call_eq(a: dict, b: dict) -> bool:
    return a["name"] == b["name"] and a["arguments"] == b["arguments"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sft_v3_scratch")
    ap.add_argument("--suite", default="in", choices=["in", "out"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)  # pointer head is optional and unused here
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    fn = "seal_tools_in_domain.json" if a.suite == "in" else "seal_tools_out_domain.json"
    rows = [normalize_example(r) for r in seal_tools_rows(ROOT / "data" / "eval" / fn)]
    # Seal ships sorted by difficulty, so any prefix is a biased sample
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

    st = collections.Counter()
    arg_err = collections.Counter()
    by_len = collections.defaultdict(collections.Counter)
    samples = []

    for ex in rows:
        pr = prompt_text(ex["query"], ex["tools"])
        pid = tok.encode(pr)
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
        spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
        gold = canon_calls(ex["answers"])
        gold_names = [c["name"] for c in gold]

        free = canon_calls(constrained_decode(model, tok, pr, ex["query"], ex["tools"], dev,
                                              name_spans=spans, gated=True))
        oracle = canon_calls(constrained_decode(model, tok, pr, ex["query"], ex["tools"], dev,
                                                name_spans=spans, gated=True,
                                                force_names=gold_names))

        n = len(gold)
        ok = free == gold
        st["n"] += 1
        st["exact"] += ok
        st["count_ok"] += len(free) == n
        st["multiset_ok"] += sorted(c["name"] for c in free) == sorted(gold_names)
        st["seq_ok"] += [c["name"] for c in free] == gold_names
        st["oracle_exact"] += oracle == gold
        by_len[n]["n"] += 1
        by_len[n]["exact"] += ok
        by_len[n]["oracle"] += oracle == gold
        by_len[n]["seq_ok"] += [c["name"] for c in free] == gold_names

        # per-call argument accuracy under the oracle: names are right by
        # construction, so every mismatch here is an argument problem
        for g, p in zip(gold, oracle):
            st["oracle_calls"] += 1
            if call_eq(g, p):
                st["oracle_calls_ok"] += 1
                continue
            gk, pk = set(g["arguments"]), set(p["arguments"])
            if gk != pk:
                arg_err["key_set"] += 1
                arg_err[f"  missing={len(gk-pk)} extra={len(pk-gk)}"] += 1
            else:
                arg_err["values_only"] += 1
                for k in gk:
                    if g["arguments"][k] != p["arguments"][k]:
                        arg_err["  bad_value"] += 1
                        if len(samples) < 40:
                            samples.append({"tool": g["name"], "key": k,
                                            "gold": g["arguments"][k],
                                            "pred": p["arguments"][k],
                                            "query": ex["query"][:160]})

    n = st["n"]
    pct = lambda k: f"{100*st[k]/n:5.1f}%"
    print(f"\n== seal-{a.suite}  ckpt={a.ckpt}  sample={n} (seed {a.seed})")
    print(f"  exact (as scored)      {pct('exact')}")
    print(f"  call-count correct     {pct('count_ok')}")
    print(f"  name multiset correct  {pct('multiset_ok')}")
    print(f"  name sequence correct  {pct('seq_ok')}")
    print(f"  ORACLE names -> exact  {pct('oracle_exact')}   <- ceiling from perfect selection")
    oc = st["oracle_calls"]
    print(f"  oracle per-call exact  {100*st['oracle_calls_ok']/max(1,oc):5.1f}%  ({st['oracle_calls_ok']}/{oc})")
    print("\n  argument failure modes (oracle pass):")
    for k, v in arg_err.most_common():
        print(f"    {k:24s} {v}")
    print("\n  by gold call count:  n / exact / seq_ok / oracle")
    for k in sorted(by_len):
        b = by_len[k]
        print(f"    {k} calls: {b['n']:4d}  {100*b['exact']/b['n']:5.1f}%  "
              f"{100*b['seq_ok']/b['n']:5.1f}%  {100*b['oracle']/b['n']:5.1f}%")

    out = ROOT / f"diag_seal_{a.suite}_{a.ckpt}.json"
    out.write_text(json.dumps({"stats": dict(st), "arg_err": dict(arg_err),
                               "by_len": {k: dict(v) for k, v in by_len.items()},
                               "samples": samples}, indent=1))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()

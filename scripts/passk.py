"""Is the Seal-Tools failure a SEARCH problem or a CAPABILITY problem?

We decode greedily: roughly twenty sequential argmax decisions for a three-call
row. Two incompatible explanations fit the 24.3% we score, and they call for
opposite fixes:

  search      the right call array is in the model's distribution, but greedy
              decoding walks off it at one decision and loses the whole row
  capability  the right call array is not in the distribution at all

Sampling k times per row and asking whether ANY sample is exactly right
separates them. pass@k >> pass@1 means the answer is reachable and beam search
/ RL / best-of-N self-distillation will convert it. pass@k ~ pass@1 means it is
not there, no amount of search recovers it, and the effort belongs in the copy
head or the tokenizer instead.

This is a diagnostic, not a score: pass@k needs the gold answer to select a
sample, so it is an upper bound no decoder can reach, not something reportable.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4")
    ap.add_argument("--split", default="in")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    from tiny_toolcall.tokenizer import BPETokenizer
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    fn = f"seal_tools_{a.split}_domain.json"
    rows = [normalize_example(r) for r in seal_tools_rows(ROOT / "data" / "eval" / fn)]
    random.Random(a.seed).shuffle(rows)  # the file ships difficulty-sorted
    rows = rows[: a.n]

    greedy_hits = 0
    any_hits = 0          # pass@(k+1): greedy or any sample
    hits_at = [0] * (a.k + 1)
    t0 = time.time()
    for i, ex in enumerate(rows):
        pr = prompt_text(ex["query"], ex["tools"])
        pid = tok.encode(pr)
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
        spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
        gold = canon_calls(ex["answers"])

        def decode(temp: float):
            return canon_calls(constrained_decode(
                model, tok, pr, ex["query"], ex["tools"], dev,
                name_spans=spans, gated=True, temp=temp))

        got = decode(0.0) == gold
        greedy_hits += got
        found = got
        for j in range(a.k):
            if not found and decode(a.temp) == gold:
                found = True
            hits_at[j + 1] += found
        any_hits += found
        if (i + 1) % 20 == 0:
            n = i + 1
            print(f"{n}/{len(rows)}  pass@1={100*greedy_hits/n:5.1f}%  "
                  f"pass@{a.k+1}={100*any_hits/n:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    n = len(rows)
    print(f"\n=== seal-{a.split}  n={n}  k={a.k}  temp={a.temp}")
    print(f"pass@1        {100*greedy_hits/n:5.1f}%   (greedy — this is the real score)")
    for j in (1, 2, 4, a.k):
        if j <= a.k:
            print(f"pass@{j+1:<8d} {100*hits_at[j]/n:5.1f}%")
    lift = (any_hits - greedy_hits) / max(1, n)
    print(f"\nheadroom from search alone: +{100*lift:.1f} points")
    print("VERDICT:", "SEARCH problem — beam/RL will convert this"
          if lift > 0.10 else "CAPABILITY problem — search cannot recover it")


if __name__ == "__main__":
    main()

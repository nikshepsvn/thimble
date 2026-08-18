"""Does grammar-constrained decoding cost us accuracy, or buy it?

DCCD (arXiv 2603.03305) argues that constrained decoding distorts a model away
from its semantic preference — a "projection tax" — and that drafting first,
unconstrained, then constraining recovers up to +24 points. Our 100.0%
well-formed / 24.3% correct signature is exactly what that paper describes.

But the premise has a precondition: the constraint has to actually be fighting
the model. Ours was trained only ever to emit this JSON, so it may already put
nearly all its mass on valid continuations, in which case there is no tax to
recover and DCCD cannot help however well it is implemented.

This measures the tax directly. Three decoders on identical rows:

  free         no grammar at all — the model runs and we parse whatever it emits
  constrained  the shipped decoder
  oracle-best  either one, whichever is right — the ceiling if we could route

If free ~ constrained, the constraint is free and DCCD is dead here. If free >
constrained, the tax is real and drafting is worth building.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.grammar import _Decoder, constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.official import mobile_actions_rows, seal_tools_rows
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls, loads_calls
from tiny_toolcall.tokenizer import BOS, BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def free_draft(model, tok, prompt: str, dev, max_tokens: int = 200) -> str:
    """Greedy generation with no grammar whatsoever."""
    dec = _Decoder(model, tok, dev)
    dec.feed_id(BOS)
    dec.feed_str(prompt)
    out: list[int] = []
    for _ in range(max_tokens):
        nxt = int(dec.next_logits().argmax().item())
        if nxt < 4:  # pad / bos / eos / unk
            break
        out.append(nxt)
        dec.feed_id(nxt)
    return tok.decode(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4")
    ap.add_argument("--suite", default="seal-in")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    ev = ROOT / "data" / "eval"
    src = {
        "seal-in": lambda: seal_tools_rows(ev / "seal_tools_in_domain.json"),
        "seal-out": lambda: seal_tools_rows(ev / "seal_tools_out_domain.json"),
        "mobile-actions": lambda: mobile_actions_rows(ev / "mobile_actions_eval.jsonl"),
    }[a.suite]()
    rows = [normalize_example(r) for r in src]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

    n = free_ok = con_ok = both = either = parse_fail = 0
    t0 = time.time()
    for ex in rows:
        pr = prompt_text(ex["query"], ex["tools"])
        pid = tok.encode(pr)
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
        spans = {nm: (x + 1, y + 1) for nm, (x, y) in s0.items()}
        gold = canon_calls(ex["answers"])

        raw = free_draft(model, tok, pr, dev)
        parsed = loads_calls(raw)
        if parsed is None:
            parse_fail += 1
            f_ok = False
        else:
            f_ok = canon_calls(parsed) == gold
        c_ok = canon_calls(constrained_decode(
            model, tok, pr, ex["query"], ex["tools"], dev,
            name_spans=spans, gated=True)) == gold

        n += 1
        free_ok += f_ok
        con_ok += c_ok
        both += f_ok and c_ok
        either += f_ok or c_ok
        if n % 50 == 0:
            print(f"{n}/{len(rows)} free={100*free_ok/n:.1f} con={100*con_ok/n:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== {a.suite}  ckpt={a.ckpt}  n={n}")
    print(f"free (no grammar)      {100*free_ok/n:5.1f}%   unparseable: {parse_fail}")
    print(f"constrained (shipped)  {100*con_ok/n:5.1f}%")
    print(f"both agree correct     {100*both/n:5.1f}%")
    print(f"either correct         {100*either/n:5.1f}%   <- ceiling if we could route")
    tax = (free_ok - con_ok) / n
    print(f"\nprojection tax: {100*tax:+.1f} points")
    print("VERDICT:", "constraint IS costing accuracy — drafting is worth building"
          if tax > 0.02 else
          "no projection tax — the model already wants this structure, DCCD cannot help")
    (ROOT / f"draft_vs_con_{a.suite}_{a.ckpt}.json").write_text(json.dumps(
        {"n": n, "free": free_ok, "constrained": con_ok, "both": both,
         "either": either, "parse_fail": parse_fail}, indent=1))


if __name__ == "__main__":
    main()

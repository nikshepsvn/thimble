"""Definitive scorecard. Every suite, ungated AND gated, full row counts.

Reporting both columns is the point: if the lexical prior is carrying a suite,
that must be visible rather than folded into one number.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch

from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.eval import predict_s2, score_predictor
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.official import mobile_actions_rows, seal_tools_rows
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls, score_rows
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["accuracy", "name_acc", "well_formed", "non_empty", "one_call", "two_plus", "refuse"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sft2")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"])); model.load_state_dict(blob["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    ev = ROOT / "data" / "eval"
    src = {
        "mobile-actions": lambda: mobile_actions_rows(ev / "mobile_actions_eval.jsonl"),
        "seal-tools-in": lambda: seal_tools_rows(ev / "seal_tools_in_domain.json"),
        "seal-tools-out": lambda: seal_tools_rows(ev / "seal_tools_out_domain.json"),
        "local-eval": lambda: [json.loads(l) for l in (ROOT/"data/seeds/local_eval.jsonl").read_text().splitlines() if l.strip()],
        "local-ood": lambda: [json.loads(l) for l in (ROOT/"data/seeds/local_ood.jsonl").read_text().splitlines() if l.strip()],
        "droidcall": lambda: [json.loads(l) for l in (ROOT/"data/eval/droidcall_test_ours.jsonl").read_text().splitlines() if l.strip()],
    }[a.suite]()
    rows = [normalize_example(r) for r in (src[: a.limit] if a.limit else src)]

    def run(gated: bool, heads: bool = True):
        out = []
        for ex in rows:
            pr = prompt_text(ex["query"], ex["tools"]); pid = tok.encode(pr)
            s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
            spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
            pred = canon_calls(constrained_decode(model, tok, pr, ex["query"], ex["tools"], dev,
                                                  name_spans=spans, gated=gated, use_name_head=heads))
            out.append({"gold": ex["answers"], "pred": pred})
        return score_rows(out)

    res = {"suite": a.suite, "n": len(rows), "ckpt": a.ckpt}
    res["s2_lexical"] = score_predictor(rows, predict_s2)
    for label, kw in (("ungated", dict(gated=False)), ("gated", dict(gated=True)),
                      ("gated_heads_off", dict(gated=True, heads=False))):
        t0 = time.time()
        res[label] = run(**kw)
        res[label]["secs"] = round(time.time() - t0)
        print(f"{a.suite:16s} {label:16s} " +
              "  ".join(f"{k}={res[label].get(k,0):.3f}" for k in KEYS), flush=True)
    (ROOT / f"final_{a.ckpt}_{a.suite}.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()

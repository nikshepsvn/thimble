"""Score a checkpoint on YOUR catalog, with the same metric as the paper tables.

    python scripts/eval_catalog.py --ckpt thimble-v6 \
        --catalog my_tools.json --gold my_eval.jsonl

    # did adapting actually help?
    python scripts/eval_catalog.py --ckpt mydomain --baseline thimble-v6 \
        --catalog my_tools.json --gold my_eval.jsonl

Metric is ordered strict exact match — a row passes only if the function names,
the call order, and every argument value match. Same scorer as final_eval.py,
so numbers here are comparable to the ones in the README.

Gold rows are JSONL: {"query": "...", "answers": [{"name": ..., "arguments": {...}}]}
An empty "answers" list is a refusal row, and you should include some: refusal
is a decision the model makes, not a fallback, and a catalog with no refusal
rows will not tell you whether it over-calls.

With --gold omitted the script just prints predictions, which is the fastest
way to eyeball a catalog before investing in an eval set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_toolcall.data import name_spans_in_prompt, normalize_example  # noqa: E402
from tiny_toolcall.grammar import constrained_decode  # noqa: E402
from tiny_toolcall.model import Config, ToolTransformer  # noqa: E402
from tiny_toolcall.render import prompt_text  # noqa: E402
from tiny_toolcall.schema import canon_calls, score_rows  # noqa: E402
from tiny_toolcall.tokenizer import BPETokenizer  # noqa: E402

KEYS = ["accuracy", "name_acc", "well_formed", "non_empty", "one_call", "two_plus", "refuse"]


def load_model(name: str, dev: torch.device) -> ToolTransformer:
    path = ROOT / "checkpoints" / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"])
    return model.to(dev).eval()


def load_rows(catalog_path: Path, gold_path: Path | None, queries: list[str]) -> list[dict]:
    """Gold file, bare --query strings, or nothing: all become scorable rows."""
    from adapt import load_catalog  # single source of truth for catalog validation

    catalog = load_catalog(catalog_path)
    rows: list[dict] = []
    if gold_path:
        for ln, line in enumerate(gold_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            ex = json.loads(line)
            if "query" not in ex:
                raise SystemExit(f"{gold_path}:{ln} has no 'query'")
            rows.append({
                "query": ex["query"],
                "tools": ex.get("tools") or catalog,
                "answers": canon_calls(ex.get("answers") or []),
            })
    for q in queries:
        rows.append({"query": q, "tools": catalog, "answers": []})
    if not rows:
        raise SystemExit("nothing to run: pass --gold or --query")
    return [normalize_example(r) for r in rows]


def run(model: ToolTransformer, tok: BPETokenizer, rows: list[dict], dev: torch.device,
        gated: bool) -> list[dict]:
    out = []
    for ex in rows:
        pr = prompt_text(ex["query"], ex["tools"])
        pid = tok.encode(pr)
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
        spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
        pred = canon_calls(constrained_decode(
            model, tok, pr, ex["query"], ex["tools"], dev, name_spans=spans, gated=gated))
        out.append({"gold": ex["answers"], "pred": pred})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a checkpoint on your own tool catalog.")
    ap.add_argument("--ckpt", default="thimble-v6")
    ap.add_argument("--catalog", required=True, type=Path)
    ap.add_argument("--gold", type=Path, default=None, help="JSONL: {query, answers}")
    ap.add_argument("--query", action="append", default=[], help="ad-hoc query (repeatable)")
    ap.add_argument("--baseline", default=None, help="second checkpoint to compare against")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ungated", action="store_true", help="disable the lexical prior")
    ap.add_argument("--show", type=int, default=0, help="print this many predictions")
    a = ap.parse_args()

    rows = load_rows(a.catalog, a.gold, a.query)
    if a.limit:
        rows = rows[: a.limit]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    scoring = a.gold is not None
    print(f"{len(rows)} rows · {a.catalog.name} · device {dev.type}"
          f"{'' if scoring else ' · no gold, predictions only'}")

    results: dict[str, dict] = {}
    for label, ckpt in [("adapted" if a.baseline else "model", a.ckpt)] + \
                       ([("baseline", a.baseline)] if a.baseline else []):
        model = load_model(ckpt, dev)
        t0 = time.time()
        pairs = run(model, tok, rows, dev, gated=not a.ungated)
        secs = round(time.time() - t0)
        if scoring:
            s = score_rows(pairs)
            s["secs"] = secs
            results[label] = s
            print(f"{label:9s} {ckpt:16s} " +
                  "  ".join(f"{k}={s.get(k, 0):.3f}" for k in KEYS))
        for ex, pr in list(zip(rows, pairs))[: a.show]:
            print(f"\n  query: {ex['query']}\n  pred : {json.dumps(pr['pred'], ensure_ascii=False)}"
                  + (f"\n  gold : {json.dumps(pr['gold'], ensure_ascii=False)}" if scoring else ""))

    if a.baseline and scoring:
        d = results["adapted"]["accuracy"] - results["baseline"]["accuracy"]
        print(f"\nadaptation delta: {d:+.3f} strict exact match "
              f"({results['baseline']['accuracy']:.3f} -> {results['adapted']['accuracy']:.3f})")
    if scoring:
        out = ROOT / f"eval_catalog_{a.ckpt}.json"
        out.write_text(json.dumps({"ckpt": a.ckpt, "n": len(rows), **results}, indent=1))
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()

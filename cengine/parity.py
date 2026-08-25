"""Parity check: C engine vs Python stack on identical inputs.

    uv run python cengine/parity.py                # demo queries
    uv run python cengine/parity.py --jsonl F -n 50  # eval rows

Prints one line per case: OK/DIFF with both outputs on mismatch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tiny_toolcall.data import name_spans_in_prompt  # noqa: E402
from tiny_toolcall.grammar import constrained_decode  # noqa: E402
from tiny_toolcall.model import Config, ToolTransformer  # noqa: E402
from tiny_toolcall.render import prompt_text  # noqa: E402
from tiny_toolcall.schema import dumps_calls  # noqa: E402
from tiny_toolcall.tokenizer import BPETokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent

DEMO_QUERIES = [
    "make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on",
    "whats the weather in berlin tomorrow",
    "set an alarm for 6:30am called gym",
    "sing me a happy birthday song",
]


def load_model():
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / "thimble-v6.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    return model, tok


def py_decode(model, tok, query, tools):
    pr = prompt_text(query, tools)
    pid = tok.encode(pr)
    s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in tools])
    spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
    calls = constrained_decode(model, tok, pr, query, tools,
                               torch.device("cpu"), name_spans=spans, gated=True)
    return dumps_calls(calls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="eval rows with query/tools fields")
    ap.add_argument("-n", type=int, default=25)
    a = ap.parse_args()

    model, tok = load_model()
    if a.jsonl and "mobile_actions" in a.jsonl:
        from tiny_toolcall.official import mobile_actions_rows
        rows = mobile_actions_rows(Path(a.jsonl))[: a.n]
        cases = [(r["query"], r["tools"]) for r in rows]
    elif a.jsonl:
        rows = [json.loads(l) for l in Path(a.jsonl).read_text().splitlines() if l][: a.n]
        cases = [(r["query"], r["tools"]) for r in rows]
    else:
        catalog = json.loads((HERE / "demo_catalog.json").read_text())
        cases = [(q, catalog) for q in DEMO_QUERIES]

    # run all cases through the C engine via a temp jsonl
    tmp = HERE / "_parity_rows.jsonl"
    tmp.write_text("\n".join(json.dumps({"query": q, "tools": t}) for q, t in cases))
    c_out = subprocess.run(
        [str(HERE / "thimble"), "-w", str(HERE / "thimble.bin"),
         "-t", str(HERE / "tokenizer.bin"), "--jsonl", str(tmp)],
        capture_output=True, text=True, check=True).stdout.splitlines()

    ok = 0
    for i, (q, tools) in enumerate(cases):
        py = py_decode(model, tok, q, tools)
        c = c_out[i] if i < len(c_out) else "<missing>"
        match = py == c
        ok += int(match)
        print(f"{'OK  ' if match else 'DIFF'} {q[:60]}")
        if not match:
            print(f"     py: {py}")
            print(f"     c : {c}")
    print(f"\n{ok}/{len(cases)} identical")
    tmp.unlink()


if __name__ == "__main__":
    main()

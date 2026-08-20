"""Thimble in 30 seconds: natural language in, validated tool calls out.

    python demo.py                       # runs the built-in examples
    python demo.py "book a table for 2 at 7pm and text Sam the address"

Uses checkpoints/thimble-v6.pt (or --ckpt NAME). The JSON below is not
post-processed model text — structure and argument keys are force-fed by the
grammar; the model only ever chooses tools, values, and when to stop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tiny_toolcall.data import name_spans_in_prompt  # noqa: E402
from tiny_toolcall.grammar import constrained_decode  # noqa: E402
from tiny_toolcall.model import Config, ToolTransformer  # noqa: E402
from tiny_toolcall.render import prompt_text  # noqa: E402
from tiny_toolcall.schema import canon_calls  # noqa: E402
from tiny_toolcall.tokenizer import BPETokenizer  # noqa: E402

CATALOG = [
    {"name": "createReservation",
     "description": "Book a table at a restaurant",
     "parameters": {"type": "object", "properties": {
         "partySize": {"type": "integer", "description": "number of people"},
         "time": {"type": "string", "description": "reservation time"},
         "restaurant": {"type": "string", "description": "restaurant name"}},
         "required": ["partySize", "time"]}},
    {"name": "sendMessage",
     "description": "Send a text message to a contact",
     "parameters": {"type": "object", "properties": {
         "contact": {"type": "string", "description": "who to message"},
         "body": {"type": "string", "description": "message content"}},
         "required": ["contact", "body"]}},
    {"name": "setAlarm",
     "description": "Set an alarm",
     "parameters": {"type": "object", "properties": {
         "time": {"type": "string", "description": "alarm time"},
         "label": {"type": "string", "description": "alarm label"}},
         "required": ["time"]}},
    {"name": "getWeather",
     "description": "Get the weather forecast for a city",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string", "description": "city name"},
         "date": {"type": "string", "description": "date, YYYY-MM-DD"}},
         "required": ["city"]}},
]

EXAMPLES = [
    "make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on",
    "whats the weather in berlin tomorrow",
    "set an alarm for 6:30am called gym",
    "sing me a happy birthday song",  # nothing applies -> refuses with []
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="your request (default: run examples)")
    ap.add_argument("--ckpt", default="thimble-v6")
    a = ap.parse_args()

    root = Path(__file__).parent
    tok = BPETokenizer.load(root / "data" / "tokenizer.json")
    blob = torch.load(root / "checkpoints" / f"{a.ckpt}.pt",
                      map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    queries = [" ".join(a.query)] if a.query else EXAMPLES
    for q in queries:
        pr = prompt_text(q, CATALOG)
        pid = tok.encode(pr)
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in CATALOG])
        spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
        calls = canon_calls(constrained_decode(
            model, tok, pr, q, CATALOG, dev, name_spans=spans, gated=True))
        print(f"\n> {q}")
        print(json.dumps(calls, indent=2) if calls else "[]  (refused: no tool applies)")


if __name__ == "__main__":
    main()

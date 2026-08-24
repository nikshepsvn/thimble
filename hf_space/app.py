"""Thimble live demo — 48M params, CPU inference, grammar-guaranteed JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tiny_toolcall.data import name_spans_in_prompt
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import canon_calls
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).parent
tok = BPETokenizer.load(ROOT / "tokenizer.json")
blob = torch.load(ROOT / "thimble-v6.pt", map_location="cpu", weights_only=False)
model = ToolTransformer(Config(**blob["cfg"]))
model.load_state_dict(blob["model"], strict=False)
model.eval()
DEV = torch.device("cpu")

DEFAULT_CATALOG = json.dumps([
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
], indent=2)


def call(query: str, catalog_json: str) -> str:
    try:
        tools = json.loads(catalog_json)
        assert isinstance(tools, list) and tools
    except Exception:
        return "catalog must be a non-empty JSON list of tool schemas"
    query = (query or "").strip()
    if not query:
        return "type a request"
    pr = prompt_text(query, tools)
    pid = tok.encode(pr)
    s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in tools])
    spans = {n: (x + 1, y + 1) for n, (x, y) in s0.items()}
    with torch.no_grad():
        calls = canon_calls(constrained_decode(
            model, tok, pr, query, tools, DEV, name_spans=spans, gated=True))
    return json.dumps(calls, indent=2) if calls else "[]   ← refused: no tool applies"


demo = gr.Interface(
    fn=call,
    inputs=[
        gr.Textbox(label="Your request",
                   placeholder="make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on"),
        gr.Code(label="Tool catalog (editable JSON — bring your own tools)",
                value=DEFAULT_CATALOG, language="json"),
    ],
    outputs=gr.Code(label="Tool calls", language="json"),
    title="🧵 Thimble — a tool-calling layer, 48M params, live on CPU",
    description=(
        "A tool-calling layer, not a language model — it does not converse or write prose, "
        "it turns a request into calls against the schemas you give it. Structure and every "
        "argument key are **grammar-guaranteed**: malformed JSON, invented parameter names, "
        "and calls to undeclared tools are unreachable on any catalog, with no training. "
        "Edit the catalog below to try your own tools. Runs here on a free CPU.\n\n"
        "[Model](https://huggingface.co/flashvenom/thimble) · "
        "[Code + full experimental record](https://github.com/nikshepsvn/thimble)"
    ),
    examples=[
        ["make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on", DEFAULT_CATALOG],
        ["whats the weather in berlin tomorrow", DEFAULT_CATALOG],
        ["set an alarm for 6:30am called gym", DEFAULT_CATALOG],
        ["sing me a happy birthday song", DEFAULT_CATALOG],
    ],
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

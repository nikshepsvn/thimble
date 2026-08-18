"""v5 round 2: glaive-sharegpt and hermes_reasoning_tool_use.

glaive brings two distinct assets:
  calls     first gpt turn is `<functioncall> {"name": ..., "arguments": '...'}`
            — not valid JSON (single-quoted inner string) but a valid Python
            literal, so ast.literal_eval parses it exactly
  refusals  58% of first responses decline with a 1-function catalog — the
            precise case our corpus has ZERO coverage of and BFCL irrelevance
            punishes at 0.0%. Real data beats the synth we were about to buy.
            Subsampled to keep the corpus-wide refusal rate near its current
            ~14%, and capped so glaive cannot dominate the refusal signal.

hermes_reason carries clean (task, tools) columns; the chain-of-thought in its
conversations is deliberately NOT extracted (selection is not our bottleneck —
the oracle ablation caps perfect selection at +3). Expect heavy dedup against
xlam; the survivors are free.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEEDS = ROOT / "data" / "seeds"

_PY_TYPE = {"str": "string", "string": "string", "int": "integer", "integer": "integer",
            "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
            "list": "array", "array": "array", "dict": "object", "object": "object"}
_CALL = re.compile(r"<functioncall>\s*(.*?)\s*(?:</functioncall>|$)", re.DOTALL)


def _norm(s: Any) -> str:
    return re.sub(r"\W+", "", str(s).lower())


def grounded(row: dict[str, Any]) -> bool:
    q = _norm(row["query"])
    for c in row["answers"]:
        for v in (c.get("arguments") or {}).values():
            if isinstance(v, bool):
                continue
            if isinstance(v, str) and len(v) > 2 and _norm(v) not in q:
                return False
            if isinstance(v, (int, float)) and _norm(v) not in q:
                return False
    return True


def _json_objects(text: str) -> list[dict]:
    """Concatenated {...}\n{...} blobs -> list of dicts (raw_decode scan)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def _clean_tool(fn: dict) -> dict | None:
    if not fn.get("name"):
        return None
    params = fn.get("parameters") or {}
    raw_props = params.get("properties")
    if raw_props is None:                     # hermes_reason style: flat dict
        raw_props = {k: v for k, v in params.items() if isinstance(v, dict)}
        required = [k for k in raw_props]     # flat style marks nothing optional
    else:
        req_raw = params.get("required")
        req_raw = req_raw if isinstance(req_raw, list) else []  # glaive: sometimes a bool
        required = [r for r in req_raw if r in raw_props]
    props = {}
    for k, s in (raw_props or {}).items():
        s = dict(s) if isinstance(s, dict) else {"description": str(s or "")}
        s["type"] = _PY_TYPE.get(str(s.get("type", "string")).lower(), "string")
        s.pop("items", None); s.pop("properties", None); s.pop("default", None)
        props[k] = s
    return {"name": fn["name"], "description": fn.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required}}


def glaive_rows(max_refusals: int = 18000) -> list[dict[str, Any]]:
    rows, dropped, refusals = [], 0, 0
    with (RAW / "glaive_sharegpt.jsonl").open() as fh:
        for line in fh:
            conv = json.loads(line).get("conversations") or []
            sys_v = next((m["value"] for m in conv if m["from"] == "system"), "")
            human = next((m["value"] for m in conv if m["from"] == "human"), "")
            gpt = next((m["value"] for m in conv if m["from"] == "gpt"), "")
            tools = [t for t in (_clean_tool(f) for f in _json_objects(sys_v)) if t]
            if not tools or not human or not gpt:
                dropped += 1
                continue
            m = _CALL.search(gpt)
            if m:
                try:
                    call = ast.literal_eval(m.group(1))
                    args_raw = call.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (ValueError, SyntaxError, json.JSONDecodeError, AttributeError):
                    dropped += 1
                    continue
                names = {t["name"] for t in tools}
                if call.get("name") not in names or not isinstance(args, dict):
                    dropped += 1
                    continue
                rows.append({"query": human.strip(), "tools": tools,
                             "answers": [{"name": call["name"], "arguments": args}],
                             "kind": "glaive", "split": "call"})
            else:
                # a decline with a small catalog — the missing refusal case
                if refusals >= max_refusals or len(tools) > 2:
                    continue
                refusals += 1
                rows.append({"query": human.strip(), "tools": tools, "answers": [],
                             "kind": "glaive", "split": "refuse"})
    print(f"glaive: kept {len(rows)} ({refusals} small-catalog refusals), dropped {dropped}")
    return rows


def hermes_reason_rows() -> list[dict[str, Any]]:
    rows, dropped = [], 0
    with (RAW / "hermes_reason.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            task = (r.get("task") or "").strip()
            raw_tools = r.get("tools")
            raw_tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            tools = [t for t in (_clean_tool(f) for f in raw_tools or []) if t]
            # the answer rides in the last gpt turn as <tool_call> json blocks
            conv = r.get("conversations") or []
            gpt = next((m["value"] for m in reversed(conv) if m.get("from") == "gpt"), "")
            blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", gpt, re.DOTALL)
            if not task or not tools or not blocks:
                dropped += 1
                continue
            names = {t["name"] for t in tools}
            calls, ok = [], True
            for b in blocks:
                try:
                    c = json.loads(b)
                except json.JSONDecodeError:
                    try:
                        c = ast.literal_eval(b)
                    except (ValueError, SyntaxError):
                        ok = False
                        break
                if not isinstance(c, dict) or c.get("name") not in names \
                        or not isinstance(c.get("arguments"), dict):
                    ok = False
                    break
                calls.append({"name": c["name"], "arguments": c["arguments"]})
            if not ok or not calls:
                dropped += 1
                continue
            rows.append({"query": task, "tools": tools, "answers": calls,
                         "kind": "hermes_reason", "split": r.get("category", "") or ""})
    print(f"hermes_reason: kept {len(rows)}, dropped {dropped}")
    return rows


def main() -> None:
    for out, fn, raw in [("glaive_sg.jsonl", glaive_rows, "glaive_sharegpt.jsonl"),
                         ("hermes_reason.jsonl", hermes_reason_rows, "hermes_reason.jsonl")]:
        if not (RAW / raw).exists():
            print(f"SKIP {out}: raw missing")
            continue
        rows = fn()
        n = len(rows)
        rows = [r for r in rows if not r["answers"] or grounded(r)]  # refusals have no values to ground
        print(f"  evidence filter: {len(rows)}/{n} kept")
        with (SEEDS / out).open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {SEEDS/out} ({len(rows)})\n", flush=True)


if __name__ == "__main__":
    main()

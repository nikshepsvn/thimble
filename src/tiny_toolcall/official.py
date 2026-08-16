"""Converters: official eval suites -> our example format ({query, tools, answers}).

Needle-faithful notes:
- Mobile Actions declares tools as null-padded union schemas; drop null props and
  lowercase the type names. Gold args are the non-null entries of tool_calls.
- The developer turn carries current date/time; datetime arguments are resolved
  against it, so it must be visible to the model (prepended to the query).
- Scoring stays ordered strict exact match via schema.score_rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TYPE_MAP = {"OBJECT": "object", "STRING": "string", "INTEGER": "integer", "NUMBER": "number", "BOOLEAN": "boolean", "ARRAY": "array"}


def _clean_schema(params: dict[str, Any] | None) -> dict[str, Any]:
    params = params or {}
    props_in = params.get("properties") or {}
    props: dict[str, Any] = {}
    for k, spec in props_in.items():
        if spec is None:
            continue
        spec = dict(spec)
        if isinstance(spec.get("type"), str):
            spec["type"] = _TYPE_MAP.get(spec["type"], spec["type"].lower())
        props[k] = spec
    req = [r for r in (params.get("required") or []) if r in props]
    return {"type": "object", "properties": props, "required": req}


def _clean_args(args: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (args or {}).items() if v is not None}


_PY_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
            "list": "array", "dict": "object"}


def seal_tools_rows(path: Path) -> list[dict[str, Any]]:
    """Seal-Tools: prompt embeds a python-literal api_list plus task_instruction;
    the gold answer is a python-literal list of {api, parameters, responses}.

    Chained calls reference earlier outputs as 'API_call_N' placeholders; those
    are kept verbatim, matching the benchmark's own strict-match convention.
    """
    import ast
    import re

    rows: list[dict[str, Any]] = []
    for item in json.loads(path.read_text()):
        conv = {c["from"]: c["value"] for c in item.get("conversations", [])}
        human, gold_txt = conv.get("human", ""), conv.get("gpt", "")
        # both markers also appear in the instruction header; anchor on the
        # assignment forms and take the trailing task_instruction = "..."
        m = re.search(r"api_list\s*=\s*(\[.*\])\s*\ntask_instruction\s*=", human, re.DOTALL)
        q = re.search(r'task_instruction\s*=\s*"(.*?)"\s*\n\s*Output\s*:', human, re.DOTALL)
        if not m or not q:
            continue
        try:
            api_list = ast.literal_eval(m.group(1))
            gold = ast.literal_eval(gold_txt.strip())
        except (ValueError, SyntaxError):
            continue
        tools = []
        for api in api_list:
            props = {}
            for pname, spec in (api.get("parameters") or {}).items():
                spec = spec or {}
                props[pname] = {
                    "type": _PY_TYPE.get(str(spec.get("type", "str")), "string"),
                    "description": spec.get("description", ""),
                }
            tools.append({
                "name": api.get("api_name", ""),
                "description": api.get("api_description", ""),
                "parameters": {"type": "object", "properties": props,
                               "required": [r for r in (api.get("required") or []) if r in props]},
            })
        answers = [{"name": c.get("api", ""), "arguments": dict(c.get("parameters") or {})}
                   for c in (gold if isinstance(gold, list) else [])]
        rows.append({"query": q.group(1), "tools": tools, "answers": answers,
                     "kind": "official", "split": path.stem})
    return rows


def mobile_actions_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tools_raw = row["tools"]
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        tools = []
        for t in tools_raw:
            fn = t.get("function") or t
            tools.append({
                "name": fn["name"],
                "description": fn.get("description", "") or "",
                "parameters": _clean_schema(fn.get("parameters")),
            })
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        context = ""
        query = ""
        answers: list[dict[str, Any]] = []
        for m in msgs:
            role = m.get("role")
            if role == "developer" and m.get("content"):
                # keep only the date/time lines; drop boilerplate
                lines = [ln for ln in m["content"].splitlines() if "date" in ln.lower() or "day of week" in ln.lower()]
                context = " ".join(lines)
            elif role == "user" and m.get("content"):
                query = m["content"]
            elif role == "assistant":
                tc = m.get("tool_calls")
                if isinstance(tc, str):
                    tc = json.loads(tc)
                for call in tc or []:
                    fn = call.get("function") or call
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = json.loads(args)
                    answers.append({"name": fn["name"], "arguments": _clean_args(args)})
        full_query = f"{context}\n{query}".strip() if context else query
        out.append({"query": full_query, "tools": tools, "answers": answers, "kind": "official", "split": "mobile_actions"})
    return out

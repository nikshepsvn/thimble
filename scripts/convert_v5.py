"""v5 corpus: convert BitAgent / Dolci / argilla-APIGen into {query, tools, answers}.

Selection rationale is in PLAN.md §3. Shapes:

  bitagent   tools & conversation are JSON STRINGS; conversation is
             [user, 'tool call'(content={name,arguments}), assistant]; tool
             schemas use key 'arguments' with {required, type 'str', description}
  dolci      messages; assistant carries `function_calls` as Python call syntax,
             one call per line; the tool schemas ride on a `functions` field
  argilla    already {query, tools, answers}; python-ish type names

All three pass the same evidence filter as every other imported source: a scalar
argument value that is not present in the query is grounds to drop the row, not
to keep most of it — wrong values are the failure we are trying to train away.
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


def _schema_from_argdict(args: dict[str, Any]) -> dict[str, Any]:
    props, req = {}, []
    for name, spec in (args or {}).items():
        spec = spec or {}
        props[name] = {"type": _PY_TYPE.get(str(spec.get("type", "str")).lower(), "string"),
                       "description": spec.get("description", "")}
        if spec.get("required"):
            req.append(name)
    return {"type": "object", "properties": props, "required": req}


def bitagent_rows() -> list[dict[str, Any]]:
    dropped = 0
    rows = []
    with (RAW / "bitagent.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            try:
                tools_raw = r["tools"]
                tools_raw = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
                conv = r["conversation"]
                conv = json.loads(conv) if isinstance(conv, str) else conv
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            query = next((t["content"] for t in conv if t.get("role") == "user"), "")
            raw_calls = [t["content"] for t in conv if t.get("role") == "tool call"]
            calls = [c for c in raw_calls if isinstance(c, dict)]
            tools = [{"name": t.get("name", ""), "description": t.get("description", ""),
                      "parameters": _schema_from_argdict(t.get("arguments"))}
                     for t in tools_raw or [] if t.get("name")]
            names = {t["name"] for t in tools}
            if (not query or not tools or not calls or len(calls) != len(raw_calls)
                    or any(c.get("name") not in names
                           or not isinstance(c.get("arguments"), dict) for c in calls)):
                dropped += 1
                continue
            rows.append({"query": query.strip(),
                         "tools": tools,
                         "answers": [{"name": c["name"], "arguments": c["arguments"]}
                                     for c in calls],
                         "kind": "bitagent", "split": ""})
    print(f"bitagent: kept {len(rows)}, dropped {dropped}")
    return rows


def _dolci_tools(msgs: list[dict]) -> list[dict[str, Any]]:
    """The schemas ride on whichever message carries a non-null `functions`."""
    src = next((m.get("functions") for m in msgs if m.get("functions")), None)
    if not src:
        return []
    if isinstance(src, str):
        try:
            src = json.loads(src)
        except json.JSONDecodeError:
            return _tools_from_defs(src)
    tools = []
    for f in src if isinstance(src, list) else []:
        fn = f.get("function") or f
        if not fn.get("name"):
            continue
        params = fn.get("parameters") or {}
        props = {}
        for k, s in (params.get("properties") or {}).items():
            # Dolci shorthand: a bare string spec IS the description
            s = dict(s) if isinstance(s, dict) else {"description": str(s or "")}
            s["type"] = _PY_TYPE.get(str(s.get("type", "string")).lower(), "string")
            s.pop("items", None); s.pop("properties", None)
            props[k] = s
        tools.append({"name": fn["name"], "description": fn.get("description", ""),
                      "parameters": {"type": "object", "properties": props,
                                     "required": [x for x in (params.get("required") or [])
                                                  if x in props]}})
    return tools


def _tools_from_defs(src: str) -> list[dict[str, Any]]:
    """Fallback: Python def-style signatures (same parser family as dria)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    tools = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        args = fn.args
        n_req = len(args.args) - len(args.defaults)
        props, req = {}, []
        for i, a in enumerate(args.args):
            t = a.annotation.id if isinstance(a.annotation, ast.Name) else "str"
            props[a.arg] = {"type": _PY_TYPE.get(t, "string"), "description": ""}
            if i < n_req:
                req.append(a.arg)
        tools.append({"name": fn.name, "description": (ast.get_docstring(fn) or "").split("\n")[0],
                      "parameters": {"type": "object", "properties": props, "required": req}})
    return tools




def _json_safe(v: Any) -> bool:
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, list):
        return all(_json_safe(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_safe(x) for k, x in v.items())
    return False

def _parse_pycalls(text: str, known: set[str]) -> list[dict[str, Any]] | None:
    """`weather.forecast_weather_api(q="Paris", days=5)` lines -> call dicts."""
    calls = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tree = ast.parse(line, mode="eval")
        except SyntaxError:
            return None
        node = tree.body
        if not isinstance(node, ast.Call):
            return None
        # dotted names arrive as Attribute chains; reassemble
        parts, f = [], node.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr); f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
        name = ".".join(reversed(parts))
        if name not in known or node.args:
            return None
        args = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            try:
                v = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                return None
            # literal_eval also yields sets/complex, which JSON cannot express
            if not _json_safe(v):
                return None
            args[kw.arg] = v
        calls.append({"name": name, "arguments": args})
    return calls or None


def dolci_rows() -> list[dict[str, Any]]:
    rows = []
    dropped = 0
    with (RAW / "dolci.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            msgs = r.get("messages")
            msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
            if not msgs:
                dropped += 1
                continue
            tools = _dolci_tools(msgs)
            query = next((m["content"] for m in msgs if m.get("role") == "user"
                          and m.get("content")), "")
            fc = next((m.get("function_calls") for m in msgs
                       if m.get("role") == "assistant" and m.get("function_calls")), None)
            if not tools or not query or not fc:
                dropped += 1
                continue
            calls = _parse_pycalls(fc, {t["name"] for t in tools})
            if calls is None:
                dropped += 1
                continue
            rows.append({"query": query.strip(), "tools": tools, "answers": calls,
                         "kind": "dolci", "split": r.get("dataset_source", "")[:40]})
    print(f"dolci: kept {len(rows)}, dropped {dropped}")
    return rows


def argilla_rows() -> list[dict[str, Any]]:
    rows = []
    dropped = 0
    with (RAW / "apigen_argilla.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            try:
                tools_raw = r["tools"]
                tools_raw = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
                answers = r["answers"]
                answers = json.loads(answers) if isinstance(answers, str) else answers
            except (json.JSONDecodeError, KeyError, TypeError):
                dropped += 1
                continue
            tools = []
            for t in tools_raw or []:
                params = t.get("parameters") or {}
                props = {}
                for k, s in params.items():
                    s = dict(s or {})
                    props[k] = {"type": _PY_TYPE.get(str(s.get("type", "str")).split(",")[0].strip().lower(), "string"),
                                "description": s.get("description", "")}
                # APIGen marks optionality via a `default` key rather than a required list
                req = [k for k, s in params.items() if "default" not in (s or {})]
                tools.append({"name": t.get("name", ""), "description": t.get("description", ""),
                              "parameters": {"type": "object", "properties": props, "required": req}})
            names = {t["name"] for t in tools}
            if (not r.get("query") or not tools or not isinstance(answers, list) or not answers
                    or any(not isinstance(c, dict) or c.get("name") not in names
                           or not isinstance(c.get("arguments"), dict) for c in answers)):
                dropped += 1
                continue
            rows.append({"query": r["query"].strip(), "tools": tools,
                         "answers": [{"name": c["name"], "arguments": c["arguments"]}
                                     for c in answers],
                         "kind": "apigen", "split": r.get("origin", "") or ""})
    print(f"argilla-apigen: kept {len(rows)}, dropped {dropped}")
    return rows


def main() -> None:
    SEEDS.mkdir(parents=True, exist_ok=True)
    for out, fn in [("bitagent.jsonl", bitagent_rows),
                    ("dolci.jsonl", dolci_rows),
                    ("apigen.jsonl", argilla_rows)]:
        if not (RAW / (out if out != "apigen.jsonl" else "apigen_argilla.jsonl")).exists():
            print(f"SKIP {out}: raw file not downloaded yet")
            continue
        rows = fn()
        n_raw = len(rows)
        rows = [r for r in rows if grounded(r)]
        print(f"  evidence filter: {len(rows)}/{n_raw} kept")
        with (SEEDS / out).open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {SEEDS/out} ({len(rows)})\n")


if __name__ == "__main__":
    main()

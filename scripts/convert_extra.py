"""Convert newly-downloaded public sets into our {query, tools, answers} format.

Two sources, chosen for what they add rather than for row count:

  dria-pythonic  Tools are declared as typed Python signatures with :param
                 docstrings, and answers are Python source. Both the schema
                 dialect and the call syntax are unlike anything already in the
                 mix, and 32.5k of its rows are parallel/multiple calls — the
                 exact shape we score worst on (Seal-Tools is 56% three-call and
                 we get 5.8% of multi-call rows right).

  hermes-fc      OpenAI-style JSON tool schemas with <tool_call> answers, i.e.
                 the mainstream dialect, which we otherwise only see via xLAM.

Deliberately skipped: glaive-v2 (schemas and calls are embedded in free text
and need regex archaeology, and its answers are mostly refusals) and
APIGen-MT (multi-turn agentic dialogues where a single-pass query has to be
fabricated from a partial conversation).

Parsing is done with `ast.parse`, never `eval`.
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

_PY_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
            "list": "array", "dict": "object", "Any": "string"}
_SCHEMA_RE = re.compile(r"<\|functions_schema\|>(.*?)<\|end_functions_schema\|>", re.DOTALL)
_PARAM_RE = re.compile(r":param\s+(\w+)\s*:\s*(.*)")


def _ann_type(node: ast.expr | None) -> str:
    """Map a Python annotation to a JSON-schema type name."""
    if node is None:
        return "string"
    if isinstance(node, ast.Name):
        return _PY_TYPE.get(node.id, "string")
    if isinstance(node, ast.Subscript):  # List[str], Dict[str, int], Optional[x]
        base = node.value.id if isinstance(node.value, ast.Name) else ""
        if base in ("List", "Sequence", "Tuple"):
            return "array"
        if base in ("Dict", "Mapping"):
            return "object"
        if base == "Optional":
            return _ann_type(node.slice)
    return "string"


def _tools_from_source(src: str) -> list[dict[str, Any]]:
    """Python `def` signatures + :param docstrings -> our tool schema."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    tools = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        doc = ast.get_docstring(fn) or ""
        descs = dict(_PARAM_RE.findall(doc))
        # summary is everything before the first :param / :return
        summary = re.split(r"\n\s*:(?:param|return|raises)", doc)[0].strip()
        args = fn.args
        n_req = len(args.args) - len(args.defaults)
        props, required = {}, []
        for i, a in enumerate(args.args):
            props[a.arg] = {"type": _ann_type(a.annotation),
                            "description": descs.get(a.arg, "").strip()}
            if i < n_req:
                required.append(a.arg)
        tools.append({"name": fn.name, "description": summary,
                      "parameters": {"type": "object", "properties": props,
                                     "required": required}})
    return tools


def _json_safe(v: Any) -> bool:
    """literal_eval also accepts Ellipsis, sets and complex, none of which
    survive a JSON round-trip; the packer would drop those rows later anyway."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, list):
        return all(_json_safe(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_safe(x) for k, x in v.items())
    return False


def _literal(node: ast.expr) -> Any:
    """Value of a call argument, or raise if it is not a JSON-expressible literal."""
    v = ast.literal_eval(node)  # raises ValueError on names/expressions
    if not _json_safe(v):
        raise ValueError("not JSON-expressible")
    return v


def _calls_in_source_order(body: str, known: set[str]) -> list[dict[str, Any]] | None:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return None
    found: list[tuple[int, int, dict[str, Any]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in known:
            return None
        if node.args:
            return None
        args: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            try:
                args[kw.arg] = _literal(kw.value)
            except (ValueError, TypeError):
                return None
        found.append((node.lineno, node.col_offset, {"name": node.func.id, "arguments": args}))
    if not found:
        return None
    found.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in found]


def _norm(s: Any) -> str:
    return re.sub(r"\W+", "", str(s).lower())


def grounded(row: dict[str, Any]) -> bool:
    """Every scalar argument value must be evidenced in the query.

    18% of dria rows fail this — the answer calls a tool with an id that appears
    nowhere in the prompt, an artifact of flattening dialogues that had earlier
    turns. teacher.py already enforces the same rule on synthesised data, and
    since wrong argument values are the single largest source of our remaining
    errors, importing rows that reward inventing them would be worse than
    importing nothing. Rows with legitimately transformed values (dates, unit
    conversions) are lost too; that is the accepted cost of a cheap test.
    """
    q = _norm(row["query"])
    for c in row["answers"]:
        for v in c["arguments"].values():
            if isinstance(v, bool):
                continue
            if isinstance(v, str) and len(v) > 2 and _norm(v) not in q:
                return False
            if isinstance(v, (int, float)) and _norm(v) not in q:
                return False
    return True


def dria_rows(keep_types: set[str]) -> list[dict[str, Any]]:
    out, seen_bad = [], 0
    with (RAW / "dria_pythonic.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") not in keep_types:
                continue
            conv = r["conversations"]
            sys_t = next((t["content"] for t in conv if t.get("role") == "system"), "")
            user = next((t["content"] for t in conv if t.get("role") == "user"), "")
            asst = next((t["content"] for t in conv if t.get("role") == "assistant"), "")
            m = _SCHEMA_RE.search(sys_t)
            src = m.group(1) if m else sys_t.split("<|functions_schema|>")[-1]
            tools = _tools_from_source(src)
            if not tools or not user:
                continue
            code = re.search(r"```(?:python)?\s*(.*?)```", asst, re.DOTALL)
            calls = _calls_in_source_order(code.group(1) if code else asst,
                                           {t["name"] for t in tools})
            if calls is None:
                seen_bad += 1
                continue
            out.append({"query": user.strip().strip('"'), "tools": tools,
                        "answers": calls, "kind": "dria", "split": r.get("type", "")})
    print(f"dria: kept {len(out)}, dropped {seen_bad} with non-literal/positional args")
    return out


def hermes_rows() -> list[dict[str, Any]]:
    """First assistant turn only, and only when it is a tool call."""
    out, dropped = [], 0
    for fname in ("hermes_fc.jsonl", "hermes_single.jsonl"):
        p = RAW / fname
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                r = json.loads(line)
                raw_tools = r.get("tools")
                if isinstance(raw_tools, str):
                    try:
                        raw_tools = json.loads(raw_tools)
                    except json.JSONDecodeError:
                        continue
                tools = []
                for t in raw_tools or []:
                    fn = t.get("function") or t
                    if not fn.get("name"):
                        continue
                    tools.append({"name": fn["name"], "description": fn.get("description", ""),
                                  "parameters": fn.get("parameters")
                                  or {"type": "object", "properties": {}, "required": []}})
                conv = r.get("conversations") or []
                query = next((t["value"] for t in conv if t.get("from") == "human"), "")
                gpt = next((t["value"] for t in conv if t.get("from") == "gpt"), "")
                blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", gpt, re.DOTALL)
                if not tools or not query or not blocks:
                    continue
                calls, ok = [], True
                names = {t["name"] for t in tools}
                for b in blocks:
                    try:
                        c = json.loads(b)
                    except json.JSONDecodeError:
                        ok = False
                        break
                    if c.get("name") not in names or not isinstance(c.get("arguments"), dict):
                        ok = False
                        break
                    calls.append({"name": c["name"], "arguments": c["arguments"]})
                if not ok or not calls:
                    dropped += 1
                    continue
                out.append({"query": query.strip(), "tools": tools, "answers": calls,
                            "kind": "hermes", "split": r.get("category", "")})
    print(f"hermes: kept {len(out)}, dropped {dropped} unparseable")
    return out


def main() -> None:
    SEEDS.mkdir(parents=True, exist_ok=True)
    jobs = [
        # step_by_step and multi_turn assume state across calls that a single-pass
        # call array cannot express, so they are left out rather than flattened
        ("dria.jsonl", lambda: dria_rows({"simple", "parallel", "multiple"})),
        ("hermes.jsonl", hermes_rows),
    ]
    for out, fn in jobs:
        rows = fn()
        n_raw = len(rows)
        rows = [r for r in rows if grounded(r)]
        print(f"  evidence filter: {len(rows)}/{n_raw} kept")
        if not rows:
            print(f"SKIP {out}: nothing converted")
            continue
        with (SEEDS / out).open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {SEEDS / out} ({len(rows)})")


if __name__ == "__main__":
    main()

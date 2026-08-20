"""BFCL v4 single-turn adapter + AST-equivalent scorer.

The 13 single-turn categories total 3,641 rows, which is the figure Needle 2
reports; their headline is the unweighted mean over those categories, so a
category with 16 rows counts as much as one with 1,053.

Two things differ from our other suites and both are handled here rather than
by relaxing our own scorer:

1. Ground truth gives a LIST of acceptable values per parameter, not one value.
   An empty string inside that list marks the parameter as optional — omitting
   it is then also correct.
2. Two categories have no ground truth at all. `irrelevance` passes only when
   the model emits no call; `relevance` passes only when it emits some call.

Call order is compared strictly, matching the ordered-exact-match rule we use
on every other suite and the one Needle states for theirs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# BFCL declares schemas with Python-ish type names and `dict` for objects
_TYPE = {"dict": "object", "float": "number", "integer": "integer", "string": "string",
         "boolean": "boolean", "array": "array", "tuple": "array", "any": "string"}

SINGLE_TURN = [
    "simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
    "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_relevance", "live_irrelevance",
]
# categories judged on whether a call is made at all, not on its contents
NO_CALL_EXPECTED = {"irrelevance", "live_irrelevance"}
ANY_CALL_EXPECTED = {"relevance", "live_relevance"}


def _clean_schema(params: dict[str, Any] | None) -> dict[str, Any]:
    params = params or {}
    props = {}
    for k, spec in (params.get("properties") or {}).items():
        spec = dict(spec or {})
        if isinstance(spec.get("type"), str):
            spec["type"] = _TYPE.get(spec["type"], "string")
        # BFCL nests item types we do not model; drop to keep the signature compact
        spec.pop("items", None)
        spec.pop("properties", None)
        props[k] = spec
    req = [r for r in (params.get("required") or []) if r in props]
    return {"type": "object", "properties": props, "required": req}


def bfcl_rows(data_dir: Path, category: str) -> list[dict[str, Any]]:
    """One BFCL category -> our {query, tools, answers, gt} rows.

    `gt` carries the raw possible-answer structure; `answers` is a single
    representative gold (first acceptable value each) so the existing renderers
    and diagnostics keep working. Only `gt` is used for scoring.
    """
    src = data_dir / f"BFCL_v4_{category}.json"
    rows_raw = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    ans_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
    answers: dict[str, list] = {}
    if ans_path.exists():
        for line in ans_path.read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                answers[a["id"]] = a.get("ground_truth", [])

    out = []
    for r in rows_raw:
        turns = r.get("question") or [[]]
        msgs = turns[0] if turns else []
        query = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user").strip()
        tools = [{"name": f.get("name", ""), "description": f.get("description", ""),
                  "parameters": _clean_schema(f.get("parameters"))}
                 for f in (r.get("function") or [])]
        gt = answers.get(r["id"], [])
        rep = []
        for call in gt:
            for name, params in call.items():
                args = {}
                for k, vals in (params or {}).items():
                    if isinstance(vals, list) and vals:
                        if vals[0] == "" and len(vals) > 1:
                            continue  # optional and omitted in the representative
                        args[k] = vals[0]
                rep.append({"name": name, "arguments": args})
        out.append({"query": query, "tools": tools, "answers": rep, "gt": gt,
                    "kind": "bfcl", "split": category, "id": r["id"]})
    return out


def _value_ok(pred: Any, acceptable: list) -> bool:
    if not isinstance(acceptable, list):
        acceptable = [acceptable]
    for a in acceptable:
        if pred == a:
            return True
        # BFCL treats 1 and 1.0 as the same answer; so does its AST checker
        if isinstance(pred, (int, float)) and isinstance(a, (int, float)) \
                and not isinstance(pred, bool) and not isinstance(a, bool) \
                and float(pred) == float(a):
            return True
        if isinstance(pred, str) and isinstance(a, str) and pred.strip() == a.strip():
            return True
    return False


def _call_ok(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    (name, params), = gold.items()
    if pred.get("name") != name:
        return False
    args = pred.get("arguments") or {}
    params = params or {}
    for k, acceptable in params.items():
        optional = isinstance(acceptable, list) and "" in acceptable
        if k not in args:
            if not optional:
                return False
            continue
        if not _value_ok(args[k], acceptable):
            return False
    # a parameter the gold never mentions is a hallucinated argument
    return all(k in params for k in args)


def score_row(pred: list[dict[str, Any]], row: dict[str, Any]) -> bool:
    cat = row["split"]
    if cat in NO_CALL_EXPECTED:
        return not pred
    if cat in ANY_CALL_EXPECTED:
        return bool(pred)
    gt = row.get("gt") or []
    if len(pred) != len(gt):
        return False
    return all(_call_ok(p, g) for p, g in zip(pred, gt))


def unweighted_mean(per_category: dict[str, float]) -> float:
    """Needle's headline: every category counts equally regardless of size."""
    vals = [v for v in per_category.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

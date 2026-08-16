"""Canonical call records and Needle-strict scoring."""

from __future__ import annotations

import json
from typing import Any


def canon_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: args[k] for k in sorted(args)}


def canon_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in calls:
        out.append({"name": c["name"], "arguments": canon_args(dict(c.get("arguments") or {}))})
    return out


def dumps_calls(calls: list[dict[str, Any]]) -> str:
    return json.dumps(canon_calls(calls), separators=(",", ":"), ensure_ascii=False)


def loads_calls(text: str) -> list[dict[str, Any]] | None:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if raw == []:
        return []
    if isinstance(raw, dict) and "name" in raw:
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            return None
        args = item.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict):
            return None
        out.append({"name": str(item["name"]), "arguments": args})
    return canon_calls(out)


def exact_match(pred: list[dict[str, Any]] | None, gold: list[dict[str, Any]]) -> bool:
    if pred is None:
        return False
    return dumps_calls(pred) == dumps_calls(gold)


def name_match(pred: list[dict[str, Any]] | None, gold: list[dict[str, Any]]) -> bool:
    if pred is None:
        return False
    return [c["name"] for c in pred] == [c["name"] for c in gold]


def score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Needle-style single-turn table: overall + 1-call + 2-call + name-acc + well-formed + refuse."""
    n = len(rows)
    if n == 0:
        return {}
    exact = name = formed = nonempty = 0
    n1 = e1 = n2 = e2 = n0 = e0 = 0
    for r in rows:
        gold = canon_calls(r["gold"])
        pred = r.get("pred")
        if pred is not None:
            pred = canon_calls(pred)
        well = pred is not None
        formed += int(well)
        nonempty += int(bool(pred))
        hit = exact_match(pred, gold)
        exact += int(hit)
        name += int(name_match(pred, gold))
        k = len(gold)
        if k == 0:
            n0 += 1
            e0 += int(hit)
        elif k == 1:
            n1 += 1
            e1 += int(hit)
        else:
            n2 += 1
            e2 += int(hit)
    return {
        "n": n,
        "accuracy": exact / n,
        "name_acc": name / n,
        "well_formed": formed / n,
        "non_empty": nonempty / n,
        "one_call": (e1 / n1) if n1 else 0.0,
        "two_plus": (e2 / n2) if n2 else 0.0,
        "refuse": (e0 / n0) if n0 else 0.0,
        "n_one": n1,
        "n_two_plus": n2,
        "n_refuse": n0,
    }

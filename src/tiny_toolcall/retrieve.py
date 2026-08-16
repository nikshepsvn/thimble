"""DTDR: retrieve from (query, calls-so-far), not from tool results we don't have."""

from __future__ import annotations

import re
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def tool_text(tool: dict[str, Any]) -> str:
    props = tool.get("parameters", {}).get("properties", {})
    return " ".join([tool.get("name", ""), tool.get("description", ""), *props.keys()])


def retrieve(query: str, tools: list[dict[str, Any]], k: int = 5, emitted: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Score tools by overlap with query plus already-emitted call names/args (DTDR ω(q, f0:t-1))."""
    bag = _tok(query)
    if emitted:
        for c in emitted:
            # arg values help chaining; the emitted NAME tokens would boost the
            # tool we just called, the opposite of what the refresh is for
            for v in (c.get("arguments") or {}).values():
                bag |= _tok(str(v))
    scored: list[tuple[float, int, dict]] = []
    used = {c["name"] for c in (emitted or [])}
    for i, t in enumerate(tools):
        words = _tok(tool_text(t))
        overlap = len(bag & words)
        bonus = 0.25 if t.get("name") in bag else 0.0
        if t.get("name") in used:
            # demote already-called tools; strong enough to shift the ranking,
            # weak enough that a dominant overlap (parallel-same) still wins
            bonus -= 1.5
        scored.append((overlap + bonus, -i, t))
    scored.sort(reverse=True)
    top = [t for _, _, t in scored[:k]]
    # always keep already-legal next names: if nothing overlaps, still return first k
    return top or tools[:k]


def s2_pick(query: str, tools: list[dict[str, Any]], top: int = 3) -> str | None:
    """Training-free S2-style selector: argmax lexical overlap over top-3 (Looking Is Not Picking baseline)."""
    cand = retrieve(query, tools, k=top)
    if not cand:
        return None
    return cand[0]["name"]

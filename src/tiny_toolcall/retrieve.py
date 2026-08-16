"""DTDR: retrieve from (query, calls-so-far), not from tool results we don't have."""

from __future__ import annotations

import re
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")
# camelCase / PascalCase names collapse to a single token when lowercased first,
# so split on case boundaries before folding: getPostmodernTheory -> get postmodern theory
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tok(text: str) -> set[str]:
    return set(_TOKEN.findall(_CAMEL.sub(" ", text).lower()))


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


def lexical_scores(
    query: str, tools: list[dict[str, Any]], emitted: list[dict[str, Any]] | None = None
) -> dict[str, float]:
    """Normalized lexical evidence per tool, in [0,1].

    Self-describing catalogs (Seal-Tools, most enterprise APIs) are nearly solved
    by string overlap; inference-style catalogs (Mobile Actions device verbs) are
    not. This score is the prior the decoder falls back on when the model itself
    is uncertain — never when it is confident.
    """
    bag = _tok(query)
    if emitted:
        for c in emitted:
            for v in (c.get("arguments") or {}).values():
                bag |= _tok(str(v))
    used = {c["name"] for c in (emitted or [])}
    raw: dict[str, float] = {}
    for t in tools:
        words = _tok(tool_text(t))
        name_words = _tok(t.get("name", ""))
        # name matches count double: a tool whose *name* echoes the query is
        # much stronger evidence than one whose description happens to overlap
        score = len(bag & words) + 2.0 * len(bag & name_words)
        if t.get("name") in used:
            score -= 1.0
        raw[t["name"]] = max(0.0, score)
    total = sum(raw.values())
    if total <= 0:
        return {k: 1.0 / max(1, len(raw)) for k in raw}
    return {k: v / total for k, v in raw.items()}


def s2_pick(query: str, tools: list[dict[str, Any]], top: int = 3) -> str | None:
    """Training-free S2-style selector: argmax lexical overlap over top-3 (Looking Is Not Picking baseline)."""
    cand = retrieve(query, tools, k=top)
    if not cand:
        return None
    return cand[0]["name"]

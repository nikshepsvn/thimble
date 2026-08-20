"""Needle-strict scoring over our own rows (and later official suites)."""

from __future__ import annotations

from typing import Any, Callable

from tiny_toolcall.retrieve import retrieve, s2_pick
from tiny_toolcall.schema import canon_calls, loads_calls, score_rows


def predict_s2(ex: dict[str, Any]) -> list[dict[str, Any]] | None:
    """S2 name baseline: pick names lexically; args empty. Must be beaten by the trained head."""
    if not ex["answers"]:
        return []
    remaining = list(ex["tools"])
    pred = []
    emitted: list[dict[str, Any]] = []
    q = ex["query"]
    for _ in ex["answers"]:
        name = s2_pick(q, remaining, top=3)
        if not name:
            break
        pred.append({"name": name, "arguments": {}})
        emitted.append(pred[-1])
        remaining = retrieve(q, ex["tools"], k=5, emitted=emitted)
    return canon_calls(pred)


def score_predictor(rows: list[dict[str, Any]], predict: Callable[[dict], list | None]) -> dict[str, float]:
    from tiny_toolcall.data import normalize_example

    scored = []
    for ex in rows:
        ex = normalize_example(ex)  # same normalization as training pack
        pred = predict(ex)
        scored.append({"gold": ex["answers"], "pred": pred})
    return score_rows(scored)


def parse_and_score(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Score rows that already have pred_text."""
    scored = []
    for ex in rows:
        pred = loads_calls(ex.get("pred_text", ""))
        scored.append({"gold": ex["answers"], "pred": pred})
    return score_rows(scored)


def make_model_predictor(model, tok, device, use_name_head: bool = True) -> Callable[[dict], list | None]:
    """Grammar-constrained predictor. use_name_head=False is the heads-off ablation."""
    from tiny_toolcall.data import name_spans_in_prompt
    from tiny_toolcall.grammar import constrained_decode
    from tiny_toolcall.render import prompt_text

    def predict(ex: dict[str, Any]) -> list[dict[str, Any]] | None:
        prompt = prompt_text(ex["query"], ex["tools"])
        p_ids = tok.encode(prompt)
        spans0 = name_spans_in_prompt(tok, prompt, p_ids, [t["name"] for t in ex["tools"]])
        spans = {n: (s + 1, e + 1) for n, (s, e) in spans0.items()}  # +1 for BOS
        calls = constrained_decode(
            model, tok, prompt, ex["query"], ex["tools"], device,
            use_name_head=use_name_head, name_spans=spans,
        )
        return canon_calls(calls)

    return predict

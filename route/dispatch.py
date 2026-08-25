"""Fast-path tool-call dispatcher over the Thimble C engine.

One long-running `thimble --serve` subprocess; each dispatch() is a single
~130ms round trip that returns complete, grammar-validated tool calls plus two
confidence signals. Gate on them and fall back to your big model when the
dispatcher abstains.

    from route.dispatch import ThimbleDispatcher

    d = ThimbleDispatcher()          # finds cengine/ binaries by default
    r = d.dispatch(query, tools)     # tools = OpenAI-style function schemas
    if r.dispatched:
        execute(r.calls)             # bypassed the LLM entirely
    else:
        run_normal_agent(query)      # abstained: low confidence

Confidence is measured, not assumed: `vlp` is the mean log-probability of the
argument-value decisions the model made freely, `margin` is the minimum
top1-top2 gap of the tool-name choice. Thresholds must be picked per catalog
from a small gold set — see sweep() and the tables in README.md. On a catalog
the model has not been adapted to, expect low coverage; that is the gate
working, not failing.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "cengine"

# Defaults from the Mobile Actions sweep (77% coverage at 98.7% precision on
# a catalog represented in training). Re-derive for your catalog with sweep().
DEFAULT_VLP = -0.002
DEFAULT_MARGIN = 0.5


@dataclass
class Dispatch:
    calls: list
    margin: float
    vlp: float
    ms: float
    dispatched: bool


class ThimbleDispatcher:
    def __init__(self, engine: Path | None = None, weights: Path | None = None,
                 tokenizer: Path | None = None,
                 vlp_threshold: float = DEFAULT_VLP,
                 margin_threshold: float = DEFAULT_MARGIN):
        engine = engine or _DEFAULT_DIR / "thimble"
        weights = weights or _DEFAULT_DIR / "thimble-q8.bin"
        tokenizer = tokenizer or _DEFAULT_DIR / "tokenizer.bin"
        self.vlp_threshold = vlp_threshold
        self.margin_threshold = margin_threshold
        self._proc = subprocess.Popen(
            [str(engine), "-w", str(weights), "-t", str(tokenizer), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        assert self._proc.stdin is not None and self._proc.stdout is not None

    def dispatch(self, query: str, tools: list[dict]) -> Dispatch:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        req = json.dumps({"query": query, "tools": tools})
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()
        r = json.loads(self._proc.stdout.readline())
        ok = r["vlp"] >= self.vlp_threshold and r["margin"] >= self.margin_threshold
        return Dispatch(calls=r["calls"], margin=r["margin"], vlp=r["vlp"],
                        ms=r["ms"], dispatched=ok)

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait(timeout=5)


def sweep(dispatcher: ThimbleDispatcher, rows: list[dict],
          thresholds=(-0.05, -0.02, -0.01, -0.005, -0.002, -0.001)) -> list[dict]:
    """Coverage/precision table for gold rows [{query, tools, answers}].

    Pick the threshold whose precision clears your bar; if no threshold does,
    this catalog is not fast-path ready — adapt the model to it first
    (scripts/adapt.py) or keep every request on the fallback path.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tiny_toolcall.schema import dumps_calls

    results = []
    for r in rows:
        d = dispatcher.dispatch(r["query"], r["tools"])
        hit = json.dumps(d.calls, separators=(",", ":"), ensure_ascii=False) == dumps_calls(r["answers"])
        results.append((d.vlp, d.margin, hit))
    table = []
    for tau in thresholds:
        picked = [h for v, m, h in results if v >= tau and m >= dispatcher.margin_threshold]
        table.append({"vlp_threshold": tau,
                      "coverage": len(picked) / len(results),
                      "precision": (sum(picked) / len(picked)) if picked else None})
    return table

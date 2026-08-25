# route — Thimble as an agent fast path

Semantic routers answer *which tool*. They cannot answer *with what
arguments* — the request still pays for an LLM call to fill the slots. This
directory is the other half: a ~130ms local dispatcher that returns **complete,
grammar-validated tool calls**, plus the two confidence signals you gate on,
so the big model is only consulted when the dispatcher abstains.

```python
from route.dispatch import ThimbleDispatcher

d = ThimbleDispatcher()                 # wraps cengine/thimble --serve
r = d.dispatch("text Sam that i'm running late", tools)
r.calls      # [{"name":"sendMessage","arguments":{"body":"i'm running late","contact":"Sam"}}]
r.dispatched # True — margins cleared the per-catalog thresholds
r.ms         # ~130
```

`langgraph_fastpath.py` shows the LangGraph node shape: synthetic `AIMessage`
with tool_calls on dispatch, `Command(goto="agent")` on abstention, and full
router metadata (scores + thresholds) for traceability.

## The gate, measured

Confidence here is a measured quantity, not a hope. Two free signals come out
of every decode: `vlp` (mean log-probability of the argument-value decisions
the model made freely) and `margin` (minimum top1−top2 gap on tool-name
choices). Sweeping `vlp` on eval rows with gold answers:

**Mobile Actions (catalog represented in training; base exact-match 85.0%, n=300):**

| vlp ≥ | dispatched | precision of dispatched |
|---:|---:|---:|
| −0.01 | 85% | 96.5% |
| −0.005 | 81% | 97.9% |
| −0.002 | 77% | **98.7%** |
| −0.001 | 65% | 99.5% |

**Seal-Tools in-domain (hard catalog; base 34.0%, n=150, seeded shuffle):**

| vlp ≥ | dispatched | precision of dispatched |
|---:|---:|---:|
| −0.01 | 31% | 80.4% |
| −0.002 | 21% | 87.1% |
| −0.001 | 17% | 88.5% |

Read the second table honestly: on a catalog the model handles poorly, the gate
*collapses coverage* rather than confidently dispatching garbage — but the
dispatched slice still doesn't reach auto-execute precision. The deployment
rule that falls out:

1. **Fast-path a catalog only after measuring it.** `sweep()` produces this
   table for your tools from a small gold set in one command.
2. If no threshold clears your precision bar, either adapt the model to the
   catalog (`scripts/adapt.py`) or keep everything on the fallback path.
3. The interesting failure mode is absent by construction: a dispatched call
   is always schema-valid — wrong is possible, malformed is not.

Name-margin alone saturates in-domain (name accuracy is ~99% there; the
failures are argument errors) — which is why the value-side signal exists and
carries the gate. Both are exposed so you can require both.

## Wire format

`thimble --serve` reads one `{"query": ..., "tools": [...]}` per stdin line
and writes one `{"ms": ..., "margin": ..., "vlp": ..., "calls": [...]}` per
line — trivially embeddable from any language, no Python required.

"""N-gram contamination firewall for imported corpora.

The original firewall drops a training row only when its normalized query
EXACTLY equals an eval query. That was adequate when every training row came
from our own generator; it is not adequate for ~900k rows of unknown provenance,
where an eval query can arrive lightly reworded, truncated, or embedded in a
longer instruction and still teach the model the answer to a test row.

Rule: a training row is contaminated if its normalized query shares any
8-word n-gram with any eval query. Eval queries shorter than 8 words fall back
to exact normalized match (a shared 5-gram from a 6-word query would fire on
ordinary phrases; exact match is the defensible line there). 8 words follows
the decontamination range used by GPT-3 (13-gram over documents) and OLMo-style
pipelines, scaled to single-sentence queries.

A source that loses more than 5% of its rows here is dropped entirely rather
than trimmed — bulk contamination means unknown provenance, and trimming what
we can see says nothing about what we cannot.

Eval-side coverage: all 2,515 firewall queries (Mobile Actions, Seal in/out,
DroidCall test, local eval/ood) plus every BFCL v4 single-turn query, which the
old list never included.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 8
DROP_SOURCE_ABOVE = 0.05

_norm = lambda q: re.sub(r"\W+", " ", q.lower()).strip()


def _grams(q: str) -> set[tuple[str, ...]]:
    w = _norm(q).split()
    return {tuple(w[i:i + N]) for i in range(len(w) - N + 1)}


def build_eval_index() -> tuple[set[tuple[str, ...]], set[str]]:
    queries: list[str] = []
    fw = ROOT / "data" / "eval_queries.txt"
    queries += [q for q in fw.read_text().splitlines() if q.strip()]
    # BFCL v4 single-turn — never previously in the firewall
    sys.path.insert(0, str(ROOT / "src"))
    from tiny_toolcall.bfcl import SINGLE_TURN, bfcl_rows
    bfcl_dir = Path("/private/tmp/claude-501/-Users-nikshepsvn-meterline/"
                    "5721e8f1-2021-4753-870d-580772eeced5/scratchpad/gorilla/"
                    "berkeley-function-call-leaderboard/bfcl_eval/data")
    if bfcl_dir.exists():
        for cat in SINGLE_TURN:
            queries += [_norm(r["query"]) for r in bfcl_rows(bfcl_dir, cat)]
    else:
        print("WARNING: BFCL data dir missing; firewall does not cover BFCL", file=sys.stderr)
    grams: set[tuple[str, ...]] = set()
    exact: set[str] = set()
    for q in queries:
        g = _grams(q)
        if g:
            grams |= g
        else:
            exact.add(_norm(q))
    print(f"eval index: {len(queries)} queries -> {len(grams)} 8-grams, {len(exact)} exact-only")
    return grams, exact


def sweep(path: Path, grams, exact) -> None:
    rows, hit = [], 0
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            q = r.get("query", "")
            g = _grams(q)
            if (g & grams) or (not g and _norm(q) in exact):
                hit += 1
                continue
            rows.append(line)
    frac = hit / max(1, hit + len(rows))
    tag = "DROP-SOURCE" if frac > DROP_SOURCE_ABOVE else "ok"
    print(f"{path.name:28s} kept {len(rows):7d}  contaminated {hit:6d} ({100*frac:5.2f}%)  {tag}")
    if frac > DROP_SOURCE_ABOVE:
        path.rename(path.with_suffix(".jsonl.quarantined"))
        return
    if hit:
        path.write_text("".join(rows))


def main() -> None:
    # NOTE: official benchmark TRAIN splits (seal_train, official_train,
    # droidcall_train_heldout) are exempt from this sweep. Verified 2026-08-19:
    # 36% of seal_train shares 8-grams with eval yet contains ZERO exact query
    # duplicates — the overlap is template phrasing inherent to how the
    # benchmark was generated, and training on the provided split is the
    # benchmark's own protocol. The exact-query firewall at mix time still
    # applies to them. This sweep is for third-party imports, where shared
    # 8-grams imply copied rows.
    grams, exact = build_eval_index()
    targets = sys.argv[1:] or ["bitagent.jsonl", "dolci.jsonl", "apigen.jsonl",
                               "dria.jsonl", "hermes.jsonl", "xlam.jsonl", "toolace.jsonl"]
    for t in targets:
        p = ROOT / "data" / "seeds" / t
        if p.exists():
            sweep(p, grams, exact)
        else:
            print(f"{t:28s} (absent)")


if __name__ == "__main__":
    main()

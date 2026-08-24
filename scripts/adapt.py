"""Adapt Thimble to your own tool catalog.

    python scripts/adapt.py --catalog my_tools.json --name mydomain

Three stages, each resumable (`--stage synth|pack|train|all`):

  synth  ask a teacher model for validated (query -> calls) rows against YOUR
         schemas; every row is checked against your parameter types and the
         evidence rule before it is kept
  pack   blend those rows with guard data and pack two splits
  train  continue from the shipped checkpoint, annealing your blend into the
         learning-rate decay phase

Why annealing rather than a from-scratch fine-tune: on the controlled twin in
RESULTS.md the same corrective corpus scored 28.4 fed from scratch and 33.1
annealed into the decay phase. Corrective data dilutes into the average when it
competes with the whole corpus, and concentrates when it arrives late.

Why guard data is not optional: `pack_anneal.py` carries the suites we already
hold precisely so the decay phase does not drift off them. Annealing purely on
your catalog trades general tool-calling competence for it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_toolcall.cli import DATA, _firewall, _read_rows  # noqa: E402
from tiny_toolcall.data import pack_examples, save_packed  # noqa: E402
from tiny_toolcall.schema import canon_calls  # noqa: E402
from tiny_toolcall.tokenizer import BPETokenizer  # noqa: E402

SEQ_LEN = 768
BASE_CKPT = "thimble-v6"

# Guard sources, tried in order; whichever exist are sampled. These keep the
# decay phase anchored to general tool-calling instead of collapsing onto one
# catalog. Caps are per-source row counts, not repeats.
GUARDS: list[tuple[str, int]] = [
    ("seeds/official_train.jsonl", 12000),
    ("seeds/droidcall_train_heldout.jsonl", 6000),
    ("seeds/seal_train.jsonl", 12000),
    ("seeds/local_train.jsonl", 12000),
    ("seeds/dolci.jsonl", 20000),
    ("seeds/xlam.jsonl", 12000),
]

TYPES = {"string", "integer", "number", "boolean", "object", "array"}


def load_catalog(path: Path) -> list[dict]:
    """Read and validate a tool catalog: a JSON list of schemas, or {"tools": [...]}."""
    raw = json.loads(path.read_text())
    tools = raw.get("tools") if isinstance(raw, dict) else raw
    if not isinstance(tools, list) or not tools:
        raise SystemExit(f"{path}: expected a non-empty list of tool schemas")
    out: list[dict] = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not t["name"].strip():
            raise SystemExit(f"{path}: tool {i} has no name")
        params = t.get("parameters") or {}
        props = params.get("properties")
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise SystemExit(f"{path}: tool {t['name']} has non-object properties")
        for k, spec in props.items():
            if not isinstance(spec, dict):
                raise SystemExit(f"{path}: {t['name']}.{k} spec must be an object")
            if spec.get("type") not in TYPES | {None}:
                raise SystemExit(f"{path}: {t['name']}.{k} has unsupported type {spec.get('type')!r}")
        req = [r for r in (params.get("required") or []) if r in props]
        out.append({
            "name": t["name"].strip(),
            "description": str(t.get("description", "")).strip(),
            "parameters": {"type": "object", "properties": props, "required": req},
        })
    if len({t["name"] for t in out}) != len(out):
        raise SystemExit(f"{path}: duplicate tool names")
    return out


def load_examples(path: Path, catalog: list[dict]) -> list[dict]:
    """Read the caller's own gold rows. Real examples outrank synthetic ones."""
    rows: list[dict] = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{ln} is not valid JSON ({e})")
        if "query" not in ex or "answers" not in ex:
            raise SystemExit(f"{path}:{ln} needs both 'query' and 'answers'")
        rows.append({
            "query": ex["query"],
            "tools": ex.get("tools") or catalog,
            "answers": canon_calls(ex["answers"]),
            "kind": "gold",
            "split": "adapt",
        })
    return rows


def stage_synth(catalog: list[dict], out: Path, n: int, concurrency: int, seed: int, max_calls: int) -> None:
    from tiny_toolcall.teacher import synth_for_catalog

    print(f"synth: {n} rows against {len(catalog)} tools -> {out}")
    stats = asyncio.run(synth_for_catalog(
        catalog, n, out, concurrency=concurrency, seed=seed, max_calls=max_calls))
    print(f"synth done: {stats}")


def _gather_guards(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for rel, cap in GUARDS:
        p = DATA / rel
        if not p.exists():
            continue
        r = _read_rows(p)
        if not r:
            continue
        r = rng.sample(r, min(cap, len(r)))
        rows.extend(r)
        print(f"  guard {Path(rel).name:34s} {len(r):7d}")
    return rows


def stage_pack(name: str, synth_path: Path, examples: list[dict], seed: int) -> None:
    """Two splits: a balanced main phase, then a catalog-heavy decay phase."""
    rng = random.Random(seed)
    mine = _read_rows(synth_path) if synth_path.exists() else []
    if not mine and not examples:
        raise SystemExit("nothing to adapt on: run the synth stage or pass --examples")
    print(f"pack: {len(mine)} synthesized + {len(examples)} supplied rows")

    guards = _gather_guards(rng)
    if not guards:
        print("  WARNING: no guard corpora found under data/seeds — the adapted model\n"
              "           will likely forget general tool-calling. Run scripts/download.py\n"
              "           and the converters first if you care about that.")

    # Decay phase: the caller's catalog dominates, guards hold the floor.
    # Supplied gold is weighted above synthetic because it is the real
    # distribution rather than a teacher's guess at it.
    anneal = mine * 3 + examples * 6 + rng.sample(guards, min(len(guards), 30000))
    # Main phase: breadth-first, the catalog present but not dominant.
    main = guards + mine + examples * 2

    tok = BPETokenizer.load(DATA / "tokenizer.json")
    for split, rows in (("main", main), ("anneal", anneal)):
        rows = _firewall(rows)
        rng.shuffle(rows)
        ids, tags, dec, kept = pack_examples(rows, tok, seq_len=SEQ_LEN)
        out = DATA / "packed" / f"{name}_{split}"
        save_packed(out, ids, tags, dec)
        (out / "rows.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in kept))
        print(f"  {split:6s} {len(rows):7d} rows -> {ids.shape} at {out}")


def stage_train(name: str, init: str, epochs: int) -> None:
    cmd = [sys.executable, "-m", "tiny_toolcall.cli", "train",
           "--name", name, "--split", f"{name}_main",
           "--anneal-data", f"{name}_anneal",
           "--init", init, "--no-warmup", "--epochs", str(epochs)]
    print("train:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env={**__import__("os").environ,
                                                   "PYTHONPATH": str(ROOT / "src")})


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapt Thimble to your own tool catalog.")
    ap.add_argument("--catalog", required=True, type=Path, help="JSON list of tool schemas")
    ap.add_argument("--name", required=True, help="name for the adapted checkpoint and packed splits")
    ap.add_argument("--examples", type=Path, default=None,
                    help="optional JSONL of your own gold rows: {query, answers}")
    ap.add_argument("--rows", type=int, default=8000, help="rows to synthesize (default 8000)")
    ap.add_argument("--stage", default="all", choices=["synth", "pack", "train", "all"])
    ap.add_argument("--init", default=BASE_CKPT, help=f"checkpoint to continue from (default {BASE_CKPT})")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-calls", type=int, default=4, help="longest call chain to synthesize")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    catalog = load_catalog(a.catalog)
    examples = load_examples(a.examples, catalog) if a.examples else []
    synth_path = DATA / "synth" / f"adapt_{a.name}.jsonl"
    print(f"catalog: {len(catalog)} tools ({', '.join(t['name'] for t in catalog[:6])}"
          f"{'...' if len(catalog) > 6 else ''})")

    if a.stage in ("synth", "all"):
        stage_synth(catalog, synth_path, a.rows, a.concurrency, a.seed, a.max_calls)
    if a.stage in ("pack", "all"):
        stage_pack(a.name, synth_path, examples, a.seed)
    if a.stage in ("train", "all"):
        stage_train(a.name, a.init, a.epochs)
        print(f"\nadapted checkpoint: checkpoints/{a.name}.pt")
        print(f"score it:  python scripts/eval_catalog.py --ckpt {a.name} "
              f"--catalog {a.catalog} --gold <your_eval.jsonl>")


if __name__ == "__main__":
    main()

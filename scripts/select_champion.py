"""Morning champion selection: re-score every candidate checkpoint on the SAME
dev rows under ONE canonical weight config, then rank.

Why this exists: each training run computes dev loss under its own loss weights,
and the RFT twin runs structure/keys at 0.1x — its dev numbers are on a
different scale than the main run's by construction. Cross-run selection
therefore needs a canonical re-scoring. Weights here are the standard config
(structure 1.0 / keys 1.5 / names 2 / values 4 / stop 6), applied identically
to every candidate.

Selection is by dev loss ONLY. The per-candidate Seal/MA probe numbers printed
by the overnight chains go in RESULTS.md for the record, but choosing the
champion by its eval score would be test-set selection across 6+ draws and
would inflate the headline by points that are not real.

Also evaluates a uniform checkpoint soup of each run's {final, devbest, ema}
as a 4th candidate per run — checkpoint averaging is a reliably free win and
dev decides whether it earned its place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_toolcall.data import load_packed
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.train import _dev_loss, loss_weights_from_cfg

ROOT = Path(__file__).resolve().parents[1]


def load_ck(path: Path) -> tuple[ToolTransformer, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = ToolTransformer(Config(**blob["cfg"]))
    m.load_state_dict(blob["model"], strict=False)
    return m, blob


def soup(paths: list[Path]) -> ToolTransformer | None:
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths if p.exists()]
    if len(blobs) < 2:
        return None
    m = ToolTransformer(Config(**blobs[0]["cfg"]))
    avg = {}
    for k, v in blobs[0]["model"].items():
        if v.dtype.is_floating_point:
            avg[k] = sum(b["model"][k].float() for b in blobs) / len(blobs)
            avg[k] = avg[k].to(v.dtype)
        else:
            avg[k] = v
    m.load_state_dict(avg, strict=False)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/pulls")
    ap.add_argument("--dev-rows", type=int, default=5000)
    ap.add_argument("--runs", nargs="+", default=["v5:A", "v5rft:B"],
                    help="run:subdir pairs, e.g. v6c:A v6s:B")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, tags, decisions = load_packed(ROOT / "data" / "packed" / "train")
    n = ids.shape[0]
    sl = slice(max(0, n - a.dev_rows), n)
    dev_ids, dev_tags = ids[sl], tags[sl]
    dev_dec = decisions[sl.start:]
    weights = loss_weights_from_cfg({}, dev)  # canonical: the standard config
    print(f"canonical dev: {dev_ids.shape[0]} rows (identical split both runs held out)")

    cand: dict[str, ToolTransformer] = {}
    base = ROOT / a.ckpt_dir
    for spec in a.runs:
        run, sub = spec.split(":")
        variants = {f"{run}": base / sub / f"{run}.pt",
                    f"{run}_devbest": base / sub / f"{run}_devbest.pt",
                    f"{run}_ema": base / sub / f"{run}_ema.pt"}
        for name, p in variants.items():
            if p.exists():
                cand[name], _ = load_ck(p)
        s = soup(list(variants.values()))
        if s is not None:
            cand[f"{run}_soup"] = s

    results = {}
    for name, m in cand.items():
        m.to(dev)
        dl = _dev_loss(m, dev_ids, dev_tags, dev_dec, weights, dev)
        m.to("cpu")
        results[name] = dl
        print(f"  {name:18s} canonical dev = {dl:.4f}")

    champ = min(results, key=lambda k: results[k])
    print(f"\nCHAMPION (by canonical dev, pre-registered rule): {champ}  ({results[champ]:.4f})")
    (ROOT / "selection.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

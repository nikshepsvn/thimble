"""Build and pack the decay-phase anneal blend (data annealing, MiniCPM/
Llama-3/OLMo-2 practice; switch point inside decay per Parmar et al. 2024).

The blend sharpens the corrective signal exactly where WSD runs realize their
accuracy — the decay phase — while keeping guard sources for the suites we
already hold so they don't drift:

  teacher_v6 x2           targeted synth (omit / distract / canon / plain)
  seal_train x6           eval-distribution anchor (contested suite)
  omission_exemplars x2   mined selective-omission rows
  official_train x3       Mobile Actions guard (banked suite)
  droidcall_train x3      DroidCall guard (banked suite)
  general sample 60k      breadth so the decay phase is not synth-only

Same firewall and packing as the main mix; packs to data/packed/anneal.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_toolcall.cli import DATA, _firewall, _read_rows  # noqa: E402
from tiny_toolcall.data import pack_examples, save_packed  # noqa: E402
from tiny_toolcall.tokenizer import BPETokenizer  # noqa: E402

SEQ_LEN = 768


def main() -> None:
    rng = random.Random(7)
    rows: list[dict] = []

    def add(path: Path, rep: int, cap: int = 0) -> None:
        r = _read_rows(path)
        if cap:
            r = rng.sample(r, min(cap, len(r)))
        rows.extend(r * rep)
        print(f"  {path.name:28s} {len(r):7d} x{rep}")

    add(DATA / "synth" / "teacher_v6.jsonl", 2)
    add(DATA / "seeds" / "seal_train.jsonl", 6)
    add(DATA / "seeds" / "omission_exemplars.jsonl", 2)
    add(DATA / "seeds" / "official_train.jsonl", 3)
    add(DATA / "seeds" / "droidcall_train_heldout.jsonl", 3)
    add(DATA / "seeds" / "dolci.jsonl", 1, cap=40000)
    add(DATA / "seeds" / "local_train.jsonl", 1, cap=20000)

    rows = _firewall(rows)
    rng.shuffle(rows)
    print(f"anneal blend: {len(rows)} rows post-firewall")

    tok = BPETokenizer.load(DATA / "tokenizer.json")
    ids, tags, dec, kept = pack_examples(rows, tok, seq_len=SEQ_LEN)
    out = DATA / "packed" / "anneal"
    save_packed(out, ids, tags, dec)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept))
    print(f"packed {ids.shape} -> {out}  (mean real len {int((ids != 0).sum(1).mean())})")


if __name__ == "__main__":
    main()

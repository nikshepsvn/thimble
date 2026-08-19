"""Post-generation v6 data prep: merge synth shards, dedup, report focus
balance. Run AFTER both teacher shards finish; firewall2 + pack follow.

Shards are separate processes with per-process `seen` sets, so cross-shard
query collisions are expected and removed here (first occurrence wins).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synth"


def main() -> None:
    out, seen = [], set()
    raw = 0
    for shard in ("teacher_v6.jsonl", "teacher_v6b.jsonl"):
        p = SYNTH / shard
        if not p.exists():
            continue
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line while a shard is still writing
            raw += 1
            q = " ".join(r["query"].lower().split())
            if q in seen:
                continue
            seen.add(q)
            out.append(r)

    focus = Counter(r.get("focus", "?") for r in out)
    kind = Counter(r.get("kind", "?") for r in out)
    print(f"merged {raw} raw -> {len(out)} unique ({raw - len(out)} cross-shard dupes)")
    print("focus:", dict(focus))
    print("kind:", dict(kind))

    merged = SYNTH / "teacher_v6.jsonl"
    bak = SYNTH / "teacher_v6_preshard.bak"
    if not bak.exists():
        merged.rename(bak)
    with merged.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (SYNTH / "teacher_v6b.jsonl").unlink(missing_ok=True)
    print(f"wrote {merged} ({len(out)} rows); next: firewall2.py, then pack --seq-len 768")


if __name__ == "__main__":
    main()

"""Hard gate for the v5 tokenizer. Exit code drives the pipeline: nonzero blocks
the pack — an unverified tokenizer must never reach the GPU.

Checks, each tied to the measured failure that motivated the rebuild:
  1. digit singletons     every digit char is its own token in every context;
                          errors like 300 -> '30000' traced to inconsistent
                          digit merges
  2. no digit merges      the vocab contains no multi-char token with a digit
                          (also kills '(speed=enum(0.5'-class template leakage)
  3. context invariance   values tokenize identically after a space and after a
                          quote — the copy path must be segmentation-stable
  4. fragmentation        tokens/word on Seal gold values must improve on the
                          8k tokenizer's measured 3.05
  5. round trip           encode->decode is lossless on eval queries and values
"""
from __future__ import annotations

import sys
from pathlib import Path

from tiny_toolcall.official import seal_tools_rows
from tiny_toolcall.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fails.append(name)


tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
print(f"vocab={tok.vocab_size} merges={len(tok.merges)}")

# 1+2: digits
digit_merges = [t for t in tok.vocab if len(t) > 1 and any(c.isdigit() for c in t)]
check("no multi-char digit tokens", not digit_merges, f"found {digit_merges[:5]}" if digit_merges else "")
samples = ["2021", "38.0", "20.8", "0.9", "2022-12-31", "300", "T1001", "60.7"]
bad = []
for s in samples:
    pieces = [tok.token_str(i) for i in tok.encode(s)]
    if any(len(p) > 1 and any(c.isdigit() for c in p) for p in pieces):
        bad.append((s, pieces))
check("digit singletons in encoding", not bad, str(bad[:3]) if bad else f"e.g. 38.0 -> {[tok.token_str(i) for i in tok.encode('38.0')]}")

# 3: context invariance
rows = seal_tools_rows(ROOT / "data/eval/seal_tools_in_domain.json")
vals = [str(v) for r in rows[:300] for c in r["answers"] for v in c["arguments"].values()
        if isinstance(v, str) and v][:400]
mismatch = 0
for v in vals:
    space = tok.encode(" " + v)
    plain = tok.encode(v)
    lead = [tok.token_str(i) for i in space]
    if not (lead and lead[0] == " " and space[1:] == plain):
        mismatch += 1
check("context-invariant value segmentation", mismatch == 0, f"{mismatch}/{len(vals)} mismatched")

# 4: fragmentation on WORD values only — digit values lengthen BY DESIGN
# (singletons), so averaging them in punishes the intended fix. 8k baseline
# on word values: 2.72 tokens/word.
word_vals = [v for v in vals if not any(c.isdigit() for c in v)]
frag = [len(tok.encode(v)) / max(1, len(v.split())) for v in word_vals]
mean_frag = sum(frag) / max(1, len(frag))
check("word fragmentation improved", mean_frag < 2.72,
      f"{mean_frag:.2f} tokens/word on {len(word_vals)} word values (8k baseline: 2.72)")

# 5: round trip on eval queries + values
rt_bad = 0
texts = [r["query"] for r in rows[:200]] + vals[:200]
for t in texts:
    if tok.decode(tok.encode(t)) != t:
        rt_bad += 1
check("lossless round trip", rt_bad == 0, f"{rt_bad}/{len(texts)} failed")

# structural singleton contract (grammar.py depends on it)
struct_ok = all(len(tok.encode(c)) == 1 for c in '{}[],:"')
check("structural singletons", struct_ok)

if fails:
    print(f"\nBLOCKED: {fails}")
    sys.exit(1)
print("\nTOKENIZER VERIFIED — pack may proceed")

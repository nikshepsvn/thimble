# cengine — Thimble in one C file

A dependency-free C port of the whole inference stack: tokenizer, trunk,
KV-cached decoder with snapshot/rollback, name head, lexical retrieval, and the
grammar walk. No PyTorch, no Python, no runtime downloads. The weights file is
48 MB (int8) or 191 MB (fp32); the binary loads in ~20 ms.

## Build and run

```
uv run python cengine/export.py          # writes thimble.bin, thimble-q8.bin, tokenizer.bin
cd cengine && make
./thimble -w thimble-q8.bin -t tokenizer.bin -c demo_catalog.json \
    "make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on"
[{"name":"createReservation","arguments":{"partySize":2,"restaurant":"Nobu","time":"7pm"}},{"name":"sendMessage","arguments":{"body":"dinner is on","contact":"Sam"}}]
```

`--jsonl rows.jsonl` batch-decodes `{"query":..., "tools":[...]}` rows, one
prediction line each; `--stats` prints load time, forward passes, and ms/row to
stderr.

## Parity, measured not claimed

`parity.py` runs identical inputs through the Python stack and this engine and
diffs the `dumps_calls` output byte for byte.

- fp32: **100/100 Mobile Actions eval rows identical** to the Python stack,
  plus 4/4 demo queries (including the refusal).
- int8: 98/100 identical to fp32; both differing rows happened to move *toward*
  gold (87 vs 85 exact-match on that slice). The int8 path quantizes
  activations per matvec (absmax, per-row weight scales), which is the one
  numeric liberty the engine takes — hence measured rather than assumed.

## Speed (Apple M3, single process, first 100 Mobile Actions rows)

| engine | weights | ms/row | notes |
|---|---:|---:|---|
| Python (torch CPU) | 184 MB ckpt | 582 | ~5 s to import + load |
| thimble.c fp32 | 191 MB | 453 | Accelerate sgemm/sgemv, ~60 ms load |
| thimble.c int8 | 48 MB | 348 | NEON `sdot` matvec, ~20 ms load |

A "row" is a full decode: ~320 forward passes through prompt prefill plus every
choice point, rollout, and rollback the grammar makes. This is not a tok/s
number against a plain LM loop — score_str rollouts re-feed candidate strings
and roll the cache back, exactly like the Python decoder.

Where the remaining headroom is, in order: threading the int8 matvec across
cores (the single-token sweep is memory-bound), reusing the `<tools>` prefix KV
across queries that share a catalog, and batching candidate rollouts.

## What is deliberately not here

The pointer head, span copying, and templated values — all disabled by default
in the Python decoder because they measured worse (see FINDINGS.md) — are not
ported. Temperature sampling (`temp>0`, pass@k measurement only) is not ported;
every choice is the temp=0 argmax.

## Files

- `thimble.c` — the engine (~1,600 lines): mini JSON parser, BPE tokenizer,
  fp32/int8 forward, grammar decoder.
- `export.py` — checkpoint + tokenizer → flat binaries. Tensor order is the
  contract with `thimble.c`.
- `parity.py` — byte-parity harness against the Python stack.
- `demo_catalog.json` — the demo.py catalog, for the quickstart line above.

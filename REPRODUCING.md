# Repository layout and reproduction

Everything needed to rebuild the published numbers from scratch. For what the
model is and how to use it, see [README.md](README.md); for what was tried and
failed, [FINDINGS.md](FINDINGS.md).

## Layout

```
src/tiny_toolcall/
  model.py       trunk + name head + pointer head (pointer disabled, see FINDINGS)
  grammar.py     constrained decoder; the five choice points
  retrieve.py    DTDR retriever + lexical prior
  render.py      compact tool signatures + per-token loss tags
  official.py    Mobile Actions / Seal-Tools adapters
  bfcl.py        BFCL v4 adapter + AST-equivalent scorer
  teacher.py     synthesis: invented catalogs, and yours (synth_for_catalog)
  train.py       Muon + AdamW, token-budget batching
scripts/
  adapt.py                 your catalog -> adapted checkpoint (synth / pack / train)
  eval_catalog.py          score any checkpoint on your catalog, vs a baseline
  seal_diag.py             oracle ablation: selection errors vs argument errors
  passk.py                 search-vs-capability diagnostic
  draft_vs_constrained.py  projection-tax measurement
  final_eval.py            full scorecard on the published suites
examples/
  catalog.json             a four-tool catalog in the expected format
  eval.jsonl               gold rows in the expected format, refusals included
```

## Reproducing the published numbers

```
uv venv && uv pip install -e .
python scripts/download.py            # public corpora + eval suites
python scripts/convert_v5.py && python scripts/convert_v5b.py
python scripts/firewall2.py           # 8-gram contamination sweep
python -m tiny_toolcall.cli pack --seq-len 768
python scripts/pack_anneal.py         # decay-phase corrective blend
python -m tiny_toolcall.cli train --name v6c --epochs 2 \
    --init <plateau-checkpoint> --no-warmup --anneal-data anneal
python scripts/final_eval.py --ckpt v6c_ema --suite seal-tools-in
```

`--seq-len 768` is not optional: the 512 default silently drops the longest rows,
which are exactly the multi-call ones (see FINDINGS.md). Champion selection is by
held-out dev loss — and see FINDINGS.md for why that selector must be
eval-distribution-flavored when recipes anneal.

Requires `.env` with `OPENROUTER_API_KEY` (synthesis) and `RUNPOD_API_KEY` (GPU),
both gitignored. Total across all three cycles: about $260; the v6 cycle alone,
which produced these numbers, about $85.


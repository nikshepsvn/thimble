# tiny-toolcall

A 44.45M-parameter factorized tool-calling model, built to test whether a very small
model can beat [Needle 2](https://github.com/cactus-compute/needle) (Cactus Compute,
Aug 2026) on its own published tables at the same size class. MIT.

It wins two of the five suites decisively and loses three. Everything below is
measured, including the losses and the ideas that did not work.

## Results

Ordered strict exact match — a row passes only if the function names, the call order,
and every argument value match. Needle 2's numbers are from their published tables.

| Suite | **tiny-toolcall** | Needle 2 (45M) | |
|---|---|---|---|
| Mobile Actions (961) | **81.5** | 63.7 | **win, +17.8** |
| DroidCall (200) | **47.5** | 17.0 | **win, 2.8x** |
| Well-formed JSON | **100.0** | 93.4 | **win, by construction** |
| Seal-Tools in-domain (700) | 24.3 | **32.6** | loss, -8.3 |
| Seal-Tools out-of-domain (654) | 18.0 | **28.7** | loss, -10.7 |
| BFCL v4 single-turn (3,641) | ~15.1 | **42.6** | loss, not close |

For scale: Needle 2 is pretrained on 115B tokens and post-trained on 38B. This model
saw **0.34B tokens total**, about **449x less**, and no pretraining phase at all.

**DroidCall caveat.** Their split script calls `random.shuffle()` unseeded, so their
exact 200 rows cannot be reproduced by anyone. Ours is a seeded split from the same
pool with those rows firewalled out of training — same methodology, different rows.

## How it works

A deep-thin gated trunk (d=448, 20 layers, GQA 8/4, SwiGLU x2.0, QK-norm, sandwich
RMSNorm, tied embeddings) plus three factorized capabilities:

1. **Retriever** — `retrieve(query, tools, emitted=...)`, a DTDR-style
   (arXiv 2512.17052) refresh conditioned on the *partial plan*. These benchmarks are
   single-pass with no tool results, so the refresh happens after each emitted call.
2. **Name head** — a bilinear readout scoring candidate tool-name spans in the prompt
   against the hidden state at the decision position, motivated by "Looking Is Not
   Picking" (arXiv 2606.16364): tool mis-selection is a readout failure, not a
   perception one. It has to beat a training-free lexical baseline to keep its
   parameters, and a heads-off decode is the standing ablation.
3. **Grammar decoder** — the tokenizer keeps JSON structural characters as singleton
   tokens, so structure is force-fed exactly and the model is consulted only at five
   choice points: refuse-vs-call, name, optional-include, value content,
   stop-vs-continue. **Argument keys are forced from the schema**, so parameter-name
   hallucination is structurally unreachable.

Training is weighted cross-entropy matched to the observed error distribution
(structure 1x, keys 1.5x, names 2x, values 4x, stop decision 6x) plus a name-head
auxiliary loss, with Muon on trunk 2D weights and AdamW on embeddings, norms and heads.

### What the grammar actually buys

Measured, not assumed. Decoding the same rows with the grammar and with no grammar at
all (`scripts/draft_vs_constrained.py`):

| suite | free generation | grammar-constrained |
|---|---|---|
| Mobile Actions (150) | 78.7 | 78.7 |
| Seal-Tools in (150) | 26.7 | 28.0 |

On Mobile Actions the two agree on **150 of 150 rows**. The grammar is not an accuracy
mechanism here — it is a *reliability* mechanism. It buys unconditional well-formedness,
unreachable parameter-name hallucination, and parseability on the ~11% of Seal rows
where free generation emits invalid JSON. That is worth having, and it is not the same
claim as "constrained decoding makes the model correct".

## What did not work

Five plausible ideas, each killed by a measurement rather than an argument. This is the
part most worth reading.

| idea | result | why |
|---|---|---|
| Word-span copying | **-30 pts** | span boundaries are wrong more often than free generation |
| Pointer/copy head (endpoints) | **-16 pts** | head learns, but its confidence does not correlate with correctness |
| `MAX_CALLS` 4 -> 6 | **0.000** | the cap was never binding; long rows fail for other reasons |
| Global optional-skip prior | rejected | inclusion rate is catalog-dependent (33% Seal, 94% Mobile Actions) — a constant trades one suite for another |
| Search: beam / RL / best-of-N | **capped at +5.0** | pass@9 is an *oracle* bound and only reaches 29.3 vs the 32.6 needed |
| Draft-then-constrain (DCCD) | **no effect possible** | requires a projection tax; measured at -1.3 and +0.0, so there is nothing to recover |

Two of these reversed conclusions we would otherwise have shipped on intuition. The
pointer-head result is independently corroborated: pointer generators are reported to
*hurt* structured extraction under grammar-constrained decoding in low-resource settings.

Needle's architecture is also not the advantage. Their own controlled study
(arXiv 2607.18363) finds deleting the FFN costs 0.47 nats at matched depth and only
breaks even at matched parameters (0.006 nats), and this model independently matches
every load-bearing item in their recipe: QK-normalization, sandwich norm, 20 layers,
Muon, and the same token-level loss weights.

## The remaining gap, precisely

Row accuracy factors as `P(name sequence) x p^n`, where `p` is per-call argument
accuracy. Measured on Seal-Tools in-domain: 24.3 at 80.4% name accuracy, which
back-solves to **p = 0.593**. Beating 32.6 requires **p = 0.687**.

| p | projected Seal-in |
|---|---|
| 0.593 | 24.3 (measured) |
| 0.65 | 28.4 |
| **0.687** | **32.6** |

Tool selection is not the constraint. The entire remaining task is +9.4 points of
per-call argument accuracy, and the dominant error classes are paraphrase-instead-of-copy
(`'round, red in color, grows rapidly'` for `'round, red, fast growing'`), argument
over-specification, and rare-word tokenizer fragmentation at an 8,192 vocabulary.

## Known defects

- **Never refuses on small catalogs.** BFCL `irrelevance` scores 0.0%. Two causes:
  `lexical_scores` normalizes to sum 1.0, so a single-tool catalog reads as maximally
  confident even at zero token overlap; and the training mix contains no refusal example
  with fewer than three tools. The first is fixed; the second needs data.
- **640-token context.** 6.2% of BFCL rows do not fit and are scored as misses.
- **Multi-call conjunction.** 3-call rows score 12.6%; BFCL `parallel_multiple` 6.0%.

## Layout

```
src/tiny_toolcall/
  model.py       trunk + name head + pointer head (pointer disabled, see above)
  grammar.py     constrained decoder; the five choice points
  retrieve.py    DTDR retriever + lexical prior
  render.py      compact tool signatures + per-token loss tags
  official.py    Mobile Actions / Seal-Tools adapters
  bfcl.py        BFCL v4 adapter + AST-equivalent scorer
  teacher.py     OpenRouter synthesis with stepwise validation
  train.py       Muon + AdamW, token-budget batching
scripts/
  seal_diag.py             oracle ablation: selection errors vs argument errors
  passk.py                 search-vs-capability diagnostic
  draft_vs_constrained.py  projection-tax measurement
  convert_extra.py         public-corpus converters + evidence filter
  bfcl_eval.py             13-category unweighted mean
  final_eval.py            full scorecard, ungated / gated / heads-off
```

`RESULTS.md` carries the full experimental record, including every negative result and
the reasoning that produced it.

## Reproducing

```
uv venv && uv pip install -e .
python scripts/download.py            # public corpora + eval suites
python scripts/convert_extra.py       # convert + evidence-filter
python -m tiny_toolcall.cli pack --seq-len 640
python -m tiny_toolcall.cli train --name v4 --epochs 3
python scripts/final_eval.py --ckpt v4 --suite seal-tools-in
```

`--seq-len 640` is not optional: the default of 512 silently drops the longest 12% of
rows, which are exactly the multi-call ones.

Requires `.env` with `OPENROUTER_API_KEY` (synthesis) and `RUNPOD_API_KEY` (GPU) — both
gitignored. Total cost of the run that produced these numbers: about $90.

## Honest summary

A 44M model that beats a 45M model by 2.8x on one suite and +17.8 on another, at 100%
well-formed against 93.4%, trained on 449x less data. It loses Seal-Tools by 8.3 and
BFCL decisively, and the reason for each loss is measured rather than guessed. Built by
one person in roughly a day with AI assistance, against a funded team's model.

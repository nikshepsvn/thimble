# Results — v2 checkpoint (2026-08-17)

44.1M params · bf16 · 178,947 training rows · one RTX 3090 · ~$43 total

All numbers below are full row counts with the final decoder. "Ungated" is the
model alone; "gated" adds a lexical prior that is weighted only when it
discriminates on that catalog *and* the model is uncertain. Both are reported
because if the prior carries a suite, that should be visible.

## Mobile Actions — google/mobile-actions eval split, 961 rows

| Model | Params | Accuracy | Name acc. | Non-empty | 1-call | 2-call |
|---|---|---|---|---|---|---|
| **Thimble (bf16)** | **44M** | **80.4** | **99.3** | **100.0** | **86.7** | **67.9** |
| LFM2.5 230M (f16) | 230M | 69.1 | 93.0 | 98.9 | 76.1 | 55.0 |
| FunctionGemma 270M (f16) | 270M | 64.0 | 87.3 | 98.9 | 73.0 | 46.2 |
| Needle 2 (CQ2-bit) | 45M | 63.7 | 98.3 | 99.4 | 71.3 | 48.4 |
| Apple FM (on-device) | ~3B | 57.6 | 94.2 | 95.5 | 64.5 | 43.8 |

Ungated, gated and heads-off all score 80.4 — identical to the decimal. On this
suite the lexical prior contributes nothing and the factorized name head
contributes nothing; the trunk plus grammar does the work.

## Seal-Tools — we lose, and by a lot

| Suite | Ungated | Gated | Needle 2 | Gap |
|---|---|---|---|---|
| in-domain (700) | 2.4 | 4.4 | **32.6** | −28.2 |
| out-of-domain (654) | 2.0 | 3.1 | **28.7** | −25.6 |

Name accuracy: 12.4 → 18.9 gated (in-domain), against Needle's 64.9. The gate
roughly doubles the score and is nowhere near enough.

**Why.** With names correct, only 8.3% of rows have every argument right. The
model copies argument spans well out of colloquial device requests and poorly out
of terse academic instructions ("Provide information on the swimming pattern of
manta rays" → `fish_type='manta ray'`). Round-2 data (34k traces, mixed naming
conventions, 42 non-device domains) moved in-domain from 1.0 to 4.4 without
closing it. This is a training-distribution gap, not a decoder gap.

## Our own splits, 400 rows each

| Split | Lexical floor | Ungated | Gated | Gated, heads off |
|---|---|---|---|---|
| in-distribution | 21.2 | — | 100.0 | 100.0 |
| unseen schemas (OOD) | 16.8 | 81.9 | **82.8** | 80.6 |

The OOD row is the name head's only positive result all night: +2.2 points over
the same trunk without it. Everywhere else the two are identical.

## Stated asymmetries

- **Precision favours us.** bf16 here; Needle reports CQ2-bit measured end-to-end
  through a shipped C++ binary with a 256-token sliding window.
- **Domain adaptation favours us on Mobile Actions.** Its public train split
  (8,693 rows, disjoint from eval) is in our mix. Seal-Tools shows what happens
  without that advantage.
- **Deployment favours Needle, enormously.** 14MB binary on an ESP32 versus our
  168MB bf16 checkpoint and no engine.

## Decoder disclosure

Reproducing these numbers requires: grammar-constrained decoding with schema-forced
argument keys, mid-generation re-retrieval between calls, value cap of 96 tokens,
periodic-repetition blocking, and the gated lexical prior (off by default in the
"ungated" column). The scorer is Needle's ordered strict exact match, unmodified.

## Not measured

DroidCall (200), BFCL v4 single-turn (3,641), ACEBench Normal. No harness exists
for these and no claims are made about them.

---

# v3 in progress — the multi-call diagnosis (2026-08-17)

## What was actually wrong with Seal-Tools

Seal-Tools in-domain is 200 single-call rows followed by 500 multi-call rows,
394 of which need **exactly three calls**. Our v2 training mix was 53% one-call,
29% two-call, **5% three-call**. We had never taught the model to chain three
calls, and it scored **0.000 on every multi-call row** — 71% of the benchmark.

(A methodological note: the file is sorted by difficulty, with row ids like
`test_in_domain-easy-0`. Every 150-row sample we took early on was the easy
single-call prefix, which is why probes reported 61% name accuracy while the
full 700-row run reported 18.9%. Sample from the whole file or not at all.)

## The intervention

| Source | Rows | Contributes |
|---|---|---|
| xLAM-60k (APIGen, execution-verified) | 60,000 | 3,605 tools |
| ToolACE | 8,697 | 14,949 tools, incl. names with spaces |
| Seal-Tools **train** split | 12,020 | 6,637 three-call chains |
| DroidCall **train** split | 10,051 | Android intents |
| Round-3 teacher (chain-weighted) | 31,348 | 34% three-call, 14% four-call |

Corpus: 3,300 → **20,376 distinct tools**; 301k → **330k packed rows**.
A retrained tokenizer cut Seal-Tools prompts 17% (`ACTION_PICK` 10 tokens → 1)
and recovered 29,650 rows that had exceeded the length budget.

Contamination was verified, not assumed: exact normalized-query overlap between
all six training sources plus 145k teacher traces and all 2,315 eval queries
found exactly **one** coincidental collision, which was removed. A firewall now
enforces this in the training pipeline.

## Mid-training result (warm checkpoint, 34% trained)

| Metric | v2 final | v3 @34% |
|---|---|---|
| Seal single-call accuracy | 15.5 | **36.2** |
| Seal single-call name acc | ~19 | **85.0** |
| Seal multi-call accuracy | **0.000** | 1.7 |
| Seal multi-call name acc | ~5 | **35.0** |
| Emitted the right *number* of calls | ~0 | **69.2** |

The stop decision — the single token choosing `,` over `]` — was the mechanism
holding multi-call at exactly zero. It is now correct on 69% of rows.

## Remaining wall

A three-call row passes only if all three names *and* all three argument sets
match. At ~70% per-call name accuracy that compounds to 34% at the sequence
level before arguments are even considered. Per-call accuracy is the frontier.

---

# Benchmark notes worth recording (2026-08-17)

## DroidCall's test split cannot be reproduced by anyone

`scripts/split_data.py` in the DroidCall repository shuffles the instruction pool
with `random.shuffle()` and **no seed**, then takes the first 200 rows as test.
The split therefore differs on every invocation. Needle 2's reported 17.0% is
measured on one such draw, and no third party — including Cactus themselves on a
re-run — can regenerate those exact rows.

Consequences we accept rather than paper over:
- We created our own seeded split (`data/eval/droidcall_test_ours.jsonl`, seed
  20260817) with the matching 9,851-row train file that excludes it.
- The models trained tonight saw the **full** pool, so they cannot be scored on
  DroidCall at all. No number is reported.
- Any future DroidCall comparison is "same methodology, different rows", and
  must be labelled that way.

## Seal-Tools scores its own paper differently than Needle does

The Seal-Tools paper reports Format ACC, Tool P/R/F1 and Parameter P/R/F1 — not
ordered strict exact match. Their finetuned LLaMA2-7B reaches Tool F1 80.25 and
Parameter F1 72.98, which is not comparable to Needle's "32.6 accuracy". Needle
re-scored the suite under their own strict metric; we follow Needle's metric so
our comparison against them is apples-to-apples, and we do not compare against
the numbers printed in the Seal-Tools paper.

Their error analysis is directly useful, though: of parameter-filling failures,
**70% are keyword-extraction failures from the query** — pulling the wrong span,
not misunderstanding the task. That is the same failure our own diagnostic found
and motivates copying argument values rather than generating them.

## Independent corroboration for span-copying

- Seal-Tools paper: 70% of parameter errors are extraction failures.
- LocalAgent (a 28M from-scratch tool-caller): "arg values must be copied, not
  generated"; uses a learned pointer/copy head, reports ~83% held-out.
- FuncBenchGen (arXiv 2509.26553): models "propagate incorrect or stale argument
  values"; restating known values lifted GPT-5 from 62.5% to 81.3%.

Measured on our own data: 90% of Seal-Tools argument values and 82% of Mobile
Actions values appear verbatim in the query.

## Negative result: word-span copying of argument values

Hypothesis: argument values should be **copied** from the query rather than
generated token by token. Three independent sources supported it (Seal-Tools'
error analysis, LocalAgent's pointer head, FuncBenchGen's stale-value finding),
and we measured that 90% of Seal-Tools gold values appear "verbatim" in the query.

Measured on Seal-Tools single-call, same checkpoint, 80 rows:

| decoding | accuracy | name acc |
|---|---|---|
| free generation | **40.0** | 83.8 |
| word-span copy | 10.0 | 85.0 |

Thirty points worse, with name accuracy unchanged — the entire loss is in the
arguments the change was meant to fix.

The flaw was in our own measurement. "Verbatim" was tested as *substring*
containment; the implementation enumerated *word-level* spans. Gold `'manta ray'`
is a substring of the query's `"manta rays"` but equals no word span of it.
Morphology, trailing punctuation and partial-word boundaries break word-span
copying on a large share of values, and candidate scoring cannot recover a span
that was never generated. Free generation emits tokens and handles these.

Span-copy is disabled by default and kept in the tree: a character-level span
enumerator or a learned pointer head may still be right. The word-span
approximation of that idea is not.

---

# v3 results (2026-08-17) — the data intervention worked

Two models trained in parallel on the expanded 20,376-tool corpus: a warm start
from v2 (301k rows, v2 tokenizer, 2 epochs) and a from-scratch run (330k rows,
retrained tokenizer, 3 epochs).

## Mobile Actions, 961 rows

| Model | Params | Accuracy | Name acc | 1-call | 2-call |
|---|---|---|---|---|---|
| **Thimble v3-scratch** | 44M | **82.6** | 99.3 | 89.5 | 68.8 |
| **Thimble v3-warm** | 44M | **82.3** | 99.1 | 89.4 | 68.2 |
| Thimble v2 | 44M | 80.4 | 99.3 | 86.7 | 67.9 |
| LFM2.5 230M (f16) | 230M | 69.1 | 93.0 | 76.1 | 55.0 |
| FunctionGemma 270M (f16) | 270M | 64.0 | 87.3 | 73.0 | 46.2 |
| Needle 2 (CQ2-bit) | 45M | 63.7 | 98.3 | 71.3 | 48.4 |
| Apple FM (on-device) | ~3B | 57.6 | 94.2 | 64.5 | 43.8 |

Adding 120k rows of foreign-domain data (xLAM, ToolACE, Seal-Tools, DroidCall)
**improved** Mobile Actions rather than diluting it: 80.4 → 82.6. Gated and
ungated are identical to the decimal — the lexical prior contributes nothing
here, so the trunk is doing the work.

## Seal-Tools (v3-warm)

| Split | v2 | **v3** | Needle 2 |
|---|---|---|---|
| in-domain (700) | 4.4 | **19.0** | 32.6 |
| out-of-domain (654) | 3.1 | **11.0** | 28.7 |

A 4.3× improvement in-domain. The components show where it came from:

| Bucket | v2 | v3 |
|---|---|---|
| single-call rows (200) | 15.5 | **52.0** |
| multi-call rows (500) | **0.000** | 5.8 |
| name accuracy (sequence) | 18.9 | **69.6** |

**Tool selection is now solved on this suite**: our 69.6% name accuracy exceeds
Needle's 64.9%. The entire remaining gap is argument precision compounding over
2–3 call chains.

## The gap, quantified

19.0% = 104 rows (single-call at 52%) + 29 rows (multi-call at 5.8%) of 700.
Beating 32.6% needs 228 rows. Matching Needle's single-call rate adds only 22, so
**73 rows must come from multi-call**, i.e. lifting that bucket from 5.8% to ~20%.

Working backwards, per-call argument accuracy is currently ~47%; ~71% is required.
That single number is the frontier. Named in order of expected value:

1. A learned pointer/copy head for arguments (trunk frozen, ~30 min to train).
   The word-span approximation of this failed for a diagnosed reason; the learned
   version can produce sub-word boundaries that word spans cannot.
2. Capacity — 44M parameters over 20,376 tools is thin for precision work.
3. More Seal-domain data — weakest lever; names already moved 18.9 → 69.6 on it.
4. Pretraining — Needle reaches 32.6 *without training on Seal-Tools at all*,
   off 153B tokens against our ~250M. The real structural difference.

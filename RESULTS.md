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

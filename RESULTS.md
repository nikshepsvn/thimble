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

## Warm start vs from scratch — the from-scratch model wins

Both trained on the same expanded corpus, in parallel on two GPUs, scored with
identical code on identical suites. The only differences: initialization, and a
tokenizer retrained on the full 20,376-tool corpus for the scratch run.

| Suite | v3-warm | **v3-scratch** | delta | Needle 2 |
|---|---|---|---|---|
| Mobile Actions (961) | 82.3 | **82.6** | +0.3 | 63.7 |
| Seal-Tools in (700) | 19.1 | **19.7** | +0.6 | 32.6 |
| — name accuracy | 70.7 | **76.4** | +5.7 | 64.9 |
| Seal-Tools out (654) | 11.0 | **14.1** | +3.1 | 28.7 |
| — name accuracy | 52.0 | **62.1** | +10.1 | 58.7 |

The margin grows with distance from the training domain: +0.3 on device actions,
+3.1 (a 28% relative gain) on held-out API domains. The warm start was anchoring
the model to device-action territory and cost the most exactly where the domain
was least familiar. The retrained tokenizer contributed too — 17% shorter
Seal-Tools prompts and 29,650 rows recovered from the length budget.

**Tool selection now beats Needle on every suite measured**: 99.3 vs 98.3
(Mobile Actions), 76.4 vs 64.9 (Seal in-domain), 62.1 vs 58.7 (Seal OOD). Every
remaining deficit is argument precision, not tool choice.

## The name head passes its kill criterion — but only off-distribution

An earlier draft of this section concluded the opposite, on Mobile Actions
evidence alone. The Seal-Tools ablations reverse it.

| Suite | heads on | heads off | head contributes |
|---|---|---|---|
| Mobile Actions (961) | 82.6 / name 99.3 | 82.6 / name 99.3 | nothing |
| Seal-Tools in (700) | **19.7** / name 76.1 | 17.1 / name 55.6 | +2.6 acc, **+20.5 name** |
| Seal-Tools out (654) | **14.2** / name 62.2 | 10.2 / name 38.8 | +4.0 acc, **+23.4 name** |

On the out-of-domain split the head is worth a **39% relative** improvement in
exact match. On Mobile Actions it is worth exactly nothing.

This is what "Looking Is Not Picking" predicts: the readout fix helps where the
model is *uncertain*. On Mobile Actions the trunk already reaches 99.3% name
accuracy and there is no headroom for a readout to recover; on unfamiliar API
catalogs there is 20+ points of it.

The lesson for evaluation practice is sharper than the result: measuring the
ablation on our strongest suite would have led us to delete a component worth
+39% relative on the suite we were failing. In-distribution ablations can be
blind to the exact contribution that matters.

## Second negative result: a learned pointer/copy head

After word-span copying failed, we trained a proper pointer head — token-level
start/end prediction over the prompt, 0.40M parameters, trunk frozen, supervision
derived from ~50k arguments whose gold value appears as a contiguous token
subsequence of the prompt. It shipped behind a two-endpoint confidence gate.

| Seal-Tools (multi-call weighted, 150 rows) | accuracy | 1-call | 2+-call |
|---|---|---|---|
| free generation | **25.3** | 56.7 | 4.4 |
| pointer head on | 9.3 | 23.3 | 0.0 |

Sixteen points worse, name accuracy unchanged (76.0 → 74.7) — the entire loss is
in the arguments the head was built to fix.

The head does learn: roughly half of training samples hit both endpoints exactly.
The problem is calibration — its confidence does not track its correctness, so a
0.55 two-endpoint gate still admits wrong spans, and a wrong span is a guaranteed
row failure where free generation would often have produced the right value.

**Both copy mechanisms failed.** "Copy, don't generate" is well supported in the
literature (Seal-Tools' own error analysis, LocalAgent's pointer head,
FuncBenchGen) and did not transfer here. The likely reason: grammar-constrained
free generation is already a strong copier at ~47% per-call argument accuracy,
and both mechanisms we built are noisier than that floor. A better-calibrated
head — batched training, confidence calibrated on a held-out split, span-level
rather than independent-endpoint scoring — may still beat it. Ours did not, and
we report that rather than tuning until it did.

---

# v4 preparation (2026-08-18) — a schema bug, not a capability gap

## The oracle ablation that redirected the whole effort

Every Seal-Tools number up to here was a single scalar that could have been
produced by very different failures. `scripts/seal_diag.py` decodes each row
twice — once normally, once with the call sequence pinned to the gold names —
so selection errors and argument errors separate:

| | forced-string schema | declared-type schema |
|---|---|---|
| exact (as scored) | 22.5 | 22.5 |
| name sequence correct | 76.0 | 78.0 |
| **oracle names -> exact** | **25.5** | **26.5** |
| oracle per-call argument exact | 44.3 | **51.8** |

Perfect tool selection buys 3 points. **Tool selection is not the bottleneck on
Seal-Tools and never was**; every remaining point is in argument values.

## The bug

`official.py` forced every Seal parameter to `type: "string"`, on the stated
grounds that Seal quotes its numerics. That was measured on a prefix of a file
that ships sorted by difficulty — the same sampling trap that produced the
bogus 61% name-accuracy probe earlier in this project. Across the full
in-domain set the declaration agrees with the gold far more often than not:

| declared -> gold | count |
|---|---|
| int -> int | 223 |
| float -> float | 202 |
| int -> str | 42 |
| float -> str | 31 |

Forcing string made all 498 numeric parameters unreachable: the grammar
force-feeds `"`, so the decoder could not emit a bare `2021` however well it was
trained. 35.6% of in-domain rows carry at least one non-string gold value, so
the rule capped accuracy at 64.4% before decoding began.

It corrupted training as well. `dumps_calls` serialises from the real value, so
4,139 of 12,020 Seal train rows presented `year=str!` in the prompt and
`"year":2021` in the target — teaching a behaviour the decoder forbids.

Fixed by honouring the declared type (`float` -> `number`, since Seal writes
whole-valued floats as ints in 42 cases):

| reachable ceiling | before | after |
|---|---|---|
| Seal in-domain | 64.4% | **91.1%** |
| Seal out-of-domain | 71.9% | **97.4%** |

Row-level accuracy is unchanged at eval time (22.5) because the model was
trained against the broken schema; per-call argument accuracy moving 44.3 ->
51.8 is the evidence that the mechanism works and that a retrain is what
converts it.

## Negative result: the optional-include prior is catalog-dependent

Key-set errors (almost all "one optional too many") are now the largest single
argument failure mode, and 77.8% of Seal gold calls use exactly the required
set — which suggests a global skip-prior on the include/skip decision. Measuring
the prior across the training mix kills the idea:

| source | optionals included in gold |
|---|---|
| Mobile Actions | 94.1% |
| xLAM | 81.5% |
| dria | 72.9% |
| DroidCall | 57.9% |
| ToolACE | 42.8% |
| Seal-Tools | 33.4% |

A constant that helps Seal would wreck Mobile Actions. This is the repetition
blocker again in a different costume, caught before it shipped rather than after
it cost 51 points. The inclusion rate has to be inferred per catalog from the
schema, which is a training problem, not a decoding constant.

## Packing regression

`cli.py pack` defaults to `--seq-len 512`; every good checkpoint was packed at
640. Repacking with the default silently dropped 47,551 rows (330,025 -> 282,474)
and the rows it drops are the longest — which on Seal-Tools means the multi-call
ones we are trying to fix. Always pass `--seq-len 640`.

## Corpus expansion

| source | rows | why |
|---|---|---|
| dria-pythonic | 40,680 | typed Python signatures + Python call syntax: a dialect absent from the mix, and 27% parallel/multiple calls |
| hermes-fc | 1,736 | mainstream OpenAI-style JSON tool schemas |
| teacher round 4 | in progress | 5,191 unseen tool names per 7.7k rows; 1.1% query duplication against the existing 145,678 |

Both new sources are filtered so that every scalar argument value is evidenced
in the query — the same rule `teacher.py` already enforces on synthesised data.
18% of dria rows fail it (they call tools with ids appearing nowhere in the
prompt, an artifact of flattened dialogues) and are dropped: wrong argument
values are our largest error source, so importing rows that reward inventing
them would be worse than importing nothing.

Deliberately excluded: glaive-v2 (schemas and calls embedded in free text,
answers mostly refusals) and APIGen-MT (multi-turn agentic dialogues where a
single-pass query would have to be fabricated from a partial conversation).

---

# v4 results (2026-08-18)

404,290 packed rows (up 22.5% from v3's 330,025), 3 epochs, 28,227 steps on a
3090. Final lm loss 0.073, name-head accuracy 0.982. Architecture and heads
unchanged from v3 on purpose, so the deltas below are attributable to data plus
the Seal schema fix and nothing else.

| Suite | v3 | **v4** | Needle 2 | |
|---|---|---|---|---|
| Mobile Actions (961) | 82.6 | **81.5** | 63.7 | win, +17.8 |
| DroidCall (200) | — | **47.5** | 17.0 | win, 2.8x |
| Seal-Tools in (700) | 19.7 | **24.3** | 32.6 | loss, -8.3 |
| Seal-Tools out (654) | 14.1 | **~18.2** | 28.7 | loss, -10.5 |
| well-formed, all suites | 100.0 | **100.0** | 93.4 | |

DroidCall carries a permanent caveat: their split script calls
`random.shuffle()` unseeded, so their 200 rows cannot be reproduced by anyone.
Ours is a seeded split from the same pool with those rows firewalled out of
training. Same methodology, different rows, and it must be labelled that way.

## What the run bought, and what it did not

The Seal schema fix was worth doing: +4.6 in-domain and +4.1 out-of-domain for
a bug fix. The 22.5% corpus expansion was neutral to slightly negative —
Mobile Actions fell 1.1, entirely in 2-call rows (68.8 -> 65.1). dria is 27%
multi-call but averages 1.5 tools per row, which is the obvious suspect: a
multi-call habit learned under thin catalogs that does not transfer to richer
ones. Down-weighting rather than dropping is the next thing to try.

## The gap is now a single number

Row accuracy factors as P(name sequence) x p^n for p = per-call argument
accuracy. Measured on v4: seal-in 24.3 at name accuracy 80.4, and the suite's
length mix, which back-solves to **p = 0.593**. The projection made before the
run said p ~ 0.60 and predicted 24-25; it landed at 24.3, so the model of this
suite is now validated rather than assumed.

| p | projected seal-in |
|---|---|
| 0.593 | 24.3 (measured) |
| 0.65 | 28.4 |
| **0.687** | **32.6 — beats Needle** |

Tool selection is not the constraint at 80.4% name accuracy. The entire
remaining task is +9.4 points of per-call argument accuracy.

## pass@k: search or capability?

Greedy decoding makes ~20 sequential argmax decisions on a three-call row, so
the 24.3 admits two readings with opposite fixes. Sampling k times per row and
asking whether any sample matches gold exactly separates them (`scripts/passk.py`,
temperature 0.8).

Result (120-row shuffled sample, stopped at 60 once the signal was flat):

| rows | pass@1 | pass@9 | lift |
|---|---|---|---|
| 20 | 35.0 | 40.0 | +5.0 |
| 40 | 32.5 | 37.5 | +5.0 |
| 60 | 30.0 | 35.0 | +5.0 |

**Identical at every checkpoint: +5.0.** pass@9 is an oracle upper bound — it
uses the gold answer to select among nine samples, which no deployable decoder
can do — so +5.0 is the ceiling on everything search-based, not an estimate of
what a real reranker would get. Seal-in 24.3 + 5.0 = 29.3 against Needle's 32.6.

**Search cannot win Seal-Tools even with a perfect selector.** Beam search, RL,
and best-of-N self-distillation are all bounded below the target, so none of them
is the path to beating Needle here. This reverses the recommendation that would
have been made from intuition — the objective mismatch between token-level CE and
row-level exact match is real, but it is not what is costing us the rows.

The remaining gap is capability: for most failing rows the correct call array is
not in the model's distribution at any temperature. Only interventions that change
what the model can produce — targeted training data, a copy mechanism, a larger
vocabulary — can close it.

## BFCL v4 single-turn (cold)

Nothing in this project was tuned against BFCL and no BFCL data is in the
training mix, so it is the only honest generalization test here. Caveat: xLAM
is 17% of the mix and was built partly to score well on BFCL, so "unseen" is
true of the rows, not of the distribution.

| category | v4 |
|---|---|
| simple_python | 25.2 |
| multiple | 22.5 |
| live_simple | 17.1 |
| simple_javascript | 14.0 |
| simple_java | 13.0 |
| parallel | 12.5 |
| parallel_multiple | 6.0 |
| irrelevance | **0.0** |

`parallel_multiple` at 6.0 is the same conjunction failure as Seal's multi-call
collapse. `irrelevance` at 0.0 has a separate and fully diagnosed cause:
`lexical_scores` normalises to sum 1.0, so a single-tool catalog reports 1.0
confidence even at zero token overlap, and the refusal override fires
unconditionally. Compounding it, the training mix contains no refusal example
with fewer than three tools — all ~10,700 refusal rows have 3+. The model has
never seen "one tool offered, it does not fit, say no". Both are cheap to fix
and neither was addressed in this run.

## Negative result: no projection tax, so DCCD does not transfer

DCCD (arXiv 2603.03305) reports up to +24 points by drafting unconstrained and
only then applying the grammar, on the argument that constrained decoding pushes
a model toward "locally valid yet semantically incorrect" trajectories. Our
100.0% well-formed / 24.3% correct signature is precisely the symptom that paper
describes, so it looked like the best free lever available.

The mechanism requires the constraint to actually be fighting the model.
`scripts/draft_vs_constrained.py` measures that directly — the shipped decoder
against pure greedy generation with no grammar at all, on identical rows:

| suite | free (no grammar) | constrained | tax | unparseable |
|---|---|---|---|---|
| Seal-Tools in (150) | 26.7 | **28.0** | -1.3 | 16 |
| Mobile Actions (150) | 78.7 | **78.7** | +0.0 | 0 |

There is no tax. The grammar is neutral on Mobile Actions and slightly positive
on Seal-Tools, so there is no lost probability mass for draft-conditioning to
recover, and no implementation of DCCD can help here. Routing perfectly between
the two decoders — an oracle we do not have — would reach 31.3 on Seal-in,
still short of Needle's 32.6.

The reason the premise fails is that DCCD's evidence comes from general models
forced into JSON, which natively prefer prose. Ours was trained only ever to
emit this JSON, so it already places nearly all its mass on valid continuations.

### What this says about grammar-constrained decoding in this project

On Mobile Actions, free and constrained agree on **150 of 150 rows**. The
grammar is not an accuracy mechanism here at all. What it buys is:

- 100.0% well-formed output against Needle's 93.4%, unconditionally
- parameter-name hallucination made structurally unreachable, since keys are
  never generated
- parseability on the ~11% of Seal rows where free generation emits invalid JSON

Those are safety and reliability properties, and they are worth having. But the
honest statement is that the grammar makes the model's output *trustworthy*, not
*correct*, and earlier framing in this document overstated its contribution to
accuracy.

---

# v5 methodology (2026-08-18, overnight) — recorded before the finals exist

Written while both runs train, so the method is on record independent of the
outcome. Corpus: 472,407 unique rows after the free-data campaign (dolci
+120,511 was the one genuine find; bitagent yielded 44 unique of 551k and
argilla-apigen 0 of 109k — re-hosts), plus 71,277 targeted synth rows, minus
5,561 synth rows our own 8-gram firewall flagged (2.16% — the import standard
applied to ourselves). Packed: 645,323 rows at seq 768.

Key mix decision from measurement: seal_train is a perfect distributional twin
of the eval set (69.0/99.8/35.1/13.6 vs 70.0/99.7/35.9/14.0 on
3+calls/camelCase/optional-inclusion/numeric-share) while the rest of the mix
teaches 62-95% optional inclusion against Seal's 36% — the aggregate corpus was
TEACHING our largest error class. seal_train x3 -> x6.

Tokenizer: 16,384 vocab, digits never merge. Verification gate: word-value
fragmentation 2.72 -> 2.34 tokens/word, digit values lengthen by design,
'electrical engineering' 6 -> 3 tokens, mean prompt 195 -> 181, six checks
passed. The gate initially BLOCKED on a mis-specified metric that averaged the
intended digit expansion into word fragmentation; the gate was fixed, not the
tokenizer, and the distinction is the point of having gates.

Two runs, one variable: v5 (standard recipe) and v5rft (grammar-forced tokens
at 0.1x — SHAD/RFT-style; forced tokens measured at 46.4% of the weighted loss
budget). Mid-training probes at step ~21k and ~28k of 64,704 (both pre-decay):
v5 seal-in 18.7 -> 20.0, MA 61.3 -> 67.3, rising on every metric; v5rft trails
its sibling on MA (-19) and seal name-sequence (-19) at matched steps while
slightly leading on seal exact — early evidence that structural tokens carry
call-SEQUENCING signal, a mechanism the RFT paper's setting would not surface.

Model: 48.12M params (16k-vocab tied embeddings add 3.67M over v4's 44.45M;
Needle 2 is 45.0M; at their own 2-bit deployment standard v5 would be 11.5MB
against their 14MB). Reported as parameter-class-matched with exact counts.

Pre-registered selection rule: champion by canonical-weights dev loss ONLY
(scripts/select_champion.py re-scores all candidates — final/devbest/EMA/soup
x both runs — under one weight config; the twin's own dev scale differs by
construction). Per-candidate eval probes are recorded but never select.
Reserve lever: scripts/mrt_finetune.py (RLOO, partial-credit reward,
leave-one-out baseline, gold-CE alpha=0.3) fires only in the 29-32.5 band.

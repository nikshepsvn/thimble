# Findings

Every idea in this project had to beat a measurement to ship. Most did not.
`RESULTS.md` is the chronological record — the raw lab notebook, ordered by
checkpoint. This file is the same content reorganized by *what it teaches*, for
anyone building a small structured-output model who would rather not rediscover
these the expensive way.

Two of these reversed conclusions that would otherwise have shipped on intuition.

---

## The one that worked: put corrective data in the decay phase

**Hypothesis.** Data synthesized against a model's diagnosed failures should be
worth more than generic data — but *when* it arrives during training should not
matter much.

**Measured.** Controlled twin runs, identical corpus, identical architecture,
identical budget:

| | Seal-Tools in-domain |
|---|---:|
| trained from scratch on the corrective mix | 28.4 |
| annealed into the LR-decay phase of a continued run | **33.1** |

**Why.** Fed from step zero, corrective rows compete with the entire corpus and
dilute into the average. Annealed into an already-trained model during
learning-rate decay — the phase where a WSD-scheduled run crystallizes — they
concentrate exactly where behavior sets. A mid-run probe isolated the data's own
contribution at **+3.3 points at constant LR**, before any decay.

**Takeaway.** Timing is a hyperparameter of your dataset, not just your
optimizer. If you have data that targets a specific weakness, spend it late.
This is the finding most likely to transfer to your project, and it is what
`scripts/adapt.py` is built around.

---

## Copying and pointing

### Word-span copying — **−30 points**

**Hypothesis.** Argument values should be *copied* from the query, not generated
token by token. Three independent sources supported this: Seal-Tools' own error
analysis (70% of parameter errors are extraction failures), LocalAgent's pointer
head, and FuncBenchGen's stale-value finding. We measured that **90% of
Seal-Tools gold values appear verbatim in the query** — the premise was solid.

**Measured.** Emitting values as word-span selections instead of free generation
cost 30 points.

**Why.** The premise was right and the mechanism was wrong. Span *boundaries*
are wrong more often than free generation is wrong about the whole value. A
model that must pick a start and an end has two chances to fail where generation
had one.

### Pointer/copy head on endpoints — **−16 points**

**Measured.** The head learns. Its accuracy is real. Its **confidence does not
correlate with correctness**, so gating on it loses 16 points.

**Why it matters.** This is the trap: a component can be genuinely learning the
task and still be useless as a decision-maker, because gating requires
calibration, not accuracy. Independently corroborated — pointer generators are
reported to *hurt* structured extraction under grammar-constrained decoding in
low-resource settings.

**Takeaway.** Before you gate on a head's confidence, measure the confidence, not
the accuracy.

---

## Search and reranking

### Beam search / RL / best-of-N — **capped at +5.0, and that is an oracle bound**

**Measured.** pass@9 — the *oracle* over nine samples, an upper bound no
practical reranker reaches — tops out at 29.3 where 32.6 was needed.

**Takeaway.** Run the oracle bound before building the reranker. If perfect
selection over your candidate set does not clear the target, the problem is
capability, not search, and no amount of decoding cleverness will close it.

### Field-set reranking (PGR-style) — **−1.4**

**Measured.** Fixed 3 rows, broke 13.

**Why.** It targeted the key-set error bucket, and by the time it ran, training
had already fixed that bucket. The idea was correct against the model of two
weeks earlier.

**Takeaway.** Re-diagnose before you implement. Fixes have a shelf life.

### Draft-then-constrain (DCCD) — **no effect was possible**

**Measured.** The technique recovers a "projection tax" — accuracy lost when
constrained decoding overrides what free generation wanted. We measured that tax
at −1.3 and +0.0 on our two suites. There was nothing to recover.

**Takeaway.** Measure the quantity a technique is designed to recover before
implementing the technique.

---

## Loss shaping

### Down-weighting grammar-forced tokens (RFT-style) — **−12 points**

**Hypothesis.** The grammar force-feeds structure and argument keys at decode
time, and those tokens consume 46.4% of the weighted loss budget. Training the
model to predict tokens it will never be asked to produce looks like pure waste.

**Measured.** Controlled twin run with both weights at 0.1: −12 points.

**Why.** Structure tokens are not noise — they carry the *call-sequencing*
signal. Learning where a call ends is how the model learns whether another one
follows.

**Takeaway.** "The decoder handles it, so the model needn't learn it" is wrong
when the forced tokens encode structure the model still has to reason about.

---

## Decoding constants

### `MAX_CALLS` 4 → 6 — **0.000 change**

The cap was never binding. Long rows fail for other reasons. A free parameter
that looks like a limit is not necessarily the limit.

### A global optional-include prior — **rejected before shipping**

Key-set errors were the largest argument failure bucket, and 77.8% of Seal gold
calls use exactly the required set — which suggests a global skip-prior. Then we
measured the inclusion rate across the mix:

| source | optionals included in gold |
|---|---:|
| Mobile Actions | 94.1% |
| xLAM | 81.5% |
| dria | 72.9% |
| DroidCall | 57.9% |
| ToolACE | 42.8% |
| Seal-Tools | 33.4% |

A constant that helps Seal-Tools would wreck Mobile Actions. **Inclusion rate is
a property of the catalog, not of tool calling.** It has to be inferred per
catalog from the schema, which makes it a training problem rather than a
decoding constant.

---

## Reinforcement learning

### MRT/RLOO fine-tuning — **mixed, then unusable**

On v5 (a conventionally decayed checkpoint): +1.9 out-of-domain, −1.0
in-domain. On v6c (the annealed checkpoint) it **diverges at every learning rate
tried** — lr 2e-5 collapsed mean reward 0.50 → −0.12 by update 200; lr 5e-6 ran
healthy to update 550 then collapsed 0.92 → 0.13 by 650.

**Reading.** Decay-phase annealing leaves the model in a sharper, more
concentrated minimum, and policy-gradient noise unravels exactly the
specialization the anneal bought.

**Takeaway.** The two techniques are in tension. If you anneal, budget for RL
not working on top of it — and if you adapt this model, do not reach for RLOO to
squeeze out the last point.

---

## Chasing noise in the benchmark

### Matching Seal-Tools' gold numeric typing — **not learnable**

Seal-Tools' gold stores numeric-typed arguments as strings 13% of the time, and
**74% of parameters are inconsistent about it**. We measured the ceiling of a
perfect per-parameter policy at 87.8% against 86.4% for a single global rule —
a 1.4-point spread that is noise, not signal.

**Takeaway.** Measure the ceiling before optimizing toward it. Some of the gap
between you and a benchmark is the benchmark.

---

## Process failures

These are not ideas, they are mistakes. They are here because they cost real
points and both are easy to repeat.

### The champion selector picked the wrong model

Pre-registered selection was held-out loss on the general training mix. It chose
the scratch twin over the annealed one by **five thousandths of a nat** — and
the scratch twin scored 28.4 against the annealed model's 33.1.

**Why.** A general-mix dev set structurally penalizes a model that just annealed
*away* from the general mix toward a target distribution. The selector was
biased against the winning recipe by construction.

**Fix.** Whenever candidate recipes diverge in training distribution, rank them
on an eval-distribution-flavored — but eval-free — split. Ranking on the mix you
deliberately trained away from measures the wrong thing.

### Silent data loss in packing

`cli.py pack` defaults to `--seq-len 512`. Every good checkpoint was packed at
640 or 768. Repacking with the default silently dropped **47,551 rows**
(330,025 → 282,474) — and the dropped rows were exactly the longest ones, which
are exactly the multi-call ones the model was weakest at.

**Takeaway.** A truncating default is a data bug that reports success. Log what
you drop.

---

*Full chronological record, including the runs these were extracted from:
[RESULTS.md](RESULTS.md).*

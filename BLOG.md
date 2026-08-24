# Tool calling is five decisions, not a generation problem

*Building a 48M-parameter tool-calling layer for $260, and measuring the exact
point where small stops working.*

Most tool-calling models are language models that write JSON. This one is not a
language model at all. It does not converse, reason, or write prose — it reads a
catalog of typed functions and a request and returns the calls to make, with the
JSON assembled around it by a compiler. Treating it as a **layer** rather than a
model is most of the result.

The claim I would defend: turning a request into calls against an API you control
is a translation problem, not a reasoning one, and it fits in 48M parameters. On
a real app-intent catalog it reaches 86.3% ordered strict exact match — every
function name, the call order, and every argument value correct — including 73.5%
on rows that need two or more calls. And on *any* catalog, with no training at
all, it cannot emit malformed JSON, invent a parameter name, or call a tool you
did not declare.

The claim I would not: that this is general. Off a familiar catalog it degrades,
and the last sections are about exactly how and why. That boundary is the most
useful thing here, so it gets a section rather than a footnote.

## Five decisions

Given a query and a catalog of typed functions, something has to decide:
(1) does any tool apply, (2) which one, (3) which optional arguments are
licensed by the query, (4) what values go in them, (5) is another call needed.
Every other token — braces, quotes, commas, and critically *every argument key* —
is determined by the schema before the model runs.

A model that generates JSON as text spends capacity learning that `{` follows
`[`. At 48M parameters you cannot afford that. So a grammar compiled from the
declared schemas force-feeds all structure, and the model is consulted only at
the five choice points. Malformed JSON, invented argument names, and calls to
nonexistent tools become unreachable rather than unlikely: 100% well-formed
output, by construction.

Measured honestly, the grammar is a *reliability* mechanism, not an accuracy
one — on Mobile Actions, free generation and constrained decoding agree on 150
of 150 rows. It is worth being precise about this, because "constrained decoding
makes the model correct" is a claim people make and it is not the one the data
supports. What the grammar buys is that the worst failure modes cannot be
expressed at all, plus parseability on the ~11% of Seal-Tools rows where free
generation emits invalid JSON.

## The equation that ran the project

Strict exact match factors cleanly: row accuracy = P(name sequence) × pⁿ, where
p is per-call argument accuracy and n the number of calls. That equation was the
project's management structure. Each cycle: measure both factors, diagnose the
binding one, attack only that, re-measure.

- **v4** (24.3 on Seal-Tools): name sequencing 80%, p = 0.59. Both weak.
- **v5** (31.4): a rebuilt 16k tokenizer with digits as singletons, 350k new
  corpus rows, and the eval-twin train split upweighted 6×. Name sequencing
  91.5% — solved. p barely moved.
- **v6** (33.1): nothing but p, attacked bucket by bucket.

The v5 failure diagnostic said: of 193 failing calls, 66 added exactly one
optional argument the query never mentioned, ~35 bound the right value to the
wrong slot, ~30 missed a canonical date format, and ~29 were unwinnable — the
benchmark's own gold stores numeric-typed arguments as strings 13% of the time,
with 74% of parameters inconsistent about it. We measured the ceiling of any
typing policy at 87.8% against 86.4% for a global rule, called it noise, and
stopped chasing it.

So the v6 data round was three synthesis modes aimed at three buckets: schemas
rich in optionals where the gold uses only the mentioned ones; queries carrying
two dates or two amounts that must land in different slots; natural-language
dates that the call must render as ISO — behind a deterministic checker so the
evidence rule never loosens. $56 of a cheap teacher model, 74,250 validated rows,
plus 39k omission exemplars mined for free from the corpus we already had. A
mid-training probe made the causality clean: **+3.3 points at constant learning
rate**, before any decay, attributable to nothing but the corrective data.

## Anneal, don't retrain — and pick your judge carefully

We trained twins on the new corpus. One from scratch. One resumed from the old
model's plateau checkpoint, with the training data *switched to a
corrective-heavy blend during the learning-rate decay* — the phase where a
WSD-trained model crystallizes. The literature (MiniCPM, Llama 3, OLMo 2)
recommends this; now there is a controlled reading of why.

| | Seal-Tools in-domain |
|---|---:|
| from scratch on the corrective mix | 28.4 |
| annealed into the LR-decay phase | **33.1** |

Same data, same architecture, same budget. Fed from step zero, corrective data
dilutes into the average; annealed into a trained model during decay, it
concentrates exactly where behavior sets. If one thing here transfers to your
project, it is this.

Which produced the most instructive failure of the whole build: the
pre-registered champion selector — held-out loss on the general training mix —
picked the scratch twin, by five thousandths of a nat. A general-mix dev set
structurally penalizes a model that just annealed *away* from the general mix
toward the target distribution. The selector was biased against the winning
recipe by construction. Both models' full tables are published, the failure is
documented, and the fix — rank candidates on an eval-flavored but eval-free
split — is one line for the next cycle.

## The graveyard is the moat

The repo's RESULTS.md records every idea that died, with the measurement that
killed it: span-copy heads (−30), pointer heads (−16), beam/RL search
(oracle-capped below target), draft-then-constrain (no projection tax to
recover), down-weighting grammar-forced tokens (−12: structure tokens carry the
call-sequencing signal), field-set reranking (−1.4: training had already eaten
its lunch), RLOO fine-tuning on the annealed checkpoint (diverges at every
learning rate — sharp minima and policy gradients do not mix).

Two of those would have shipped on intuition. The pointer-head result is the one
I'd flag for anyone building in this space: the head *learns*, its accuracy is
real, and its confidence simply does not correlate with correctness — so gating
on it costs 16 points. Pointer generators are independently reported to hurt
structured extraction under grammar-constrained decoding in low-resource
settings. The discipline that nothing ships without an A/B on a suite it wasn't
designed for is, as far as I can tell, the actual moat.

## Where it breaks

The interesting half. Everything above describes a model operating on catalogs
it has seen. Change that one variable and the result comes apart in a way that
is measurable *inside a single benchmark*:

| Seal-Tools, same model, same metric | |
|---|---:|
| in-domain catalogs | 33.1 |
| out-of-domain catalogs | 28.1 |

Name-sequence accuracy tracks it exactly: 88% in-domain, 79% out. Every
out-of-domain deficit traces to that one number — the model is not failing to
extract arguments, it is failing to pick the right function from an unfamiliar
catalog.

Push further out and it gets worse in specific, nameable ways. On BFCL v4
single-turn the model scores 23.5. The breakdown says why, and it is not uniform
weakness: `simple_python` 29.3, but `simple_java` 14.0 and `simple_javascript`
8.0 — Java and JS schema dialects are simply absent from a deliberately
extractive ~1B-token corpus. Parallel calls collapse outright: `parallel` 12.0,
`live_parallel` 0.0. Multi-call composition works when the calls are
sequentially motivated by the query and not when they are parallel
instantiations of one schema. And 151 of 3,641 rows (4.1%) overflow the
768-token context and score as misses.

That is a data-volume boundary, not a cleverness boundary. This model saw ~1B
unique tokens with no pretraining phase at all, in a corpus deliberately spent
on depth. Breadth — Java and JS conventions, values the query paraphrases rather
than states, open-domain coverage — is what more tokens buy. Models in this
parameter class trained on two orders of magnitude more data do hold those
numbers, which is the useful control: the architecture is not the limit, the
corpus is. Crossing it is a ~$500 GPU-and-tokens project, not an idea.

**Concentration beats volume exactly where the test is narrow, and nowhere
else.** That is the honest shape of the result, and for anyone shipping an
assistant against forty endpoints they control, the narrow case is the one they
have.

## Making it yours

All of which points at the thing the model is actually for. If accuracy depends
this sharply on catalog familiarity, then the useful artifact is not the
checkpoint — it is the loop that re-specializes it.

At 48M parameters that loop is cheap enough to be routine. `scripts/adapt.py`
takes your tool schemas, has a teacher model write validated (query → calls)
rows against them, blends those with guard corpora so the model does not forget
general tool calling, and anneals the blend into the learning-rate decay phase of
a continued run from the shipped checkpoint. A few hours on one GPU, roughly $60
of synthesis. `scripts/eval_catalog.py` scores the result on your own gold rows
against the unadapted baseline, so the question "did that help?" has a number.

Nobody fine-tunes a 7B model per customer. At 48M you can, and that — rather
than any benchmark row — is the argument for building at this size.

The honest caveat: the recipe is measured, but the ergonomics are new. No
third-party catalog has been through it and published yet.

## Numbers, disclosures, receipts

48.12M parameters, fp32 checkpoint (~11.5MB at 2-bit, ~92MB at bf16 — though
what ships is the 184MB fp32 file and there is no on-device engine yet).
768-token context. Mobile Actions' public train split, 8,693 rows disjoint from
eval, is in the training mix — that is what "known catalog" means and it is the
intended operating condition, but it should be read alongside the numbers rather
than discovered later. Every training row passed an 8-gram contamination
firewall against every evaluation query of every reported suite. One model, one
row, no per-suite checkpoint shopping. The full experimental record — including
the twin that dev loss wrongly crowned — is in the repo.

*Built by one person and an AI assistant in about a week of evenings.
Everything is MIT. The failures are the useful part.*

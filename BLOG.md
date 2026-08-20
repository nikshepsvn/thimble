# Beating a 153B-token model on its own benchmark with $260 and a spreadsheet of failures

Two weeks ago Cactus released [Needle 2](https://cactuscompute.com/needle), a
45M-parameter tool-calling model that fits in 14MB and runs on a microcontroller.
It trained on 153 billion tokens. It scores 32.6% on Seal-Tools, the strict
exact-match suite its evaluation is built around, 63.7% on Google's Mobile
Actions, and it is very good.

We built a 48M model that saw about 1 billion tokens — 150x less — and beats it
on three of its five published tables, under its own metric, unmodified:

| Suite | ours | Needle 2 |
|---|---|---|
| **Seal-Tools in-domain** | **33.1** | 32.6 |
| **Mobile Actions** | **86.3** | 63.7 |
| **DroidCall** | **52.5** | 17.0 |
| Seal-Tools out-of-domain | 28.1 | **28.7** |
| BFCL single-turn | 23.5 | **42.6** |

The Seal margin is inside sampling noise and we say so. BFCL is not close and we
say that too — their 153B tokens buy a breadth our corpus deliberately traded
away, and the last section is about exactly where that boundary sits. Total
spend across every experiment, every failed idea, and every GPU-hour: about
$260. The winning cycle alone: $85.

This post is about the two ideas that did the work: a decoder that only asks the
model five questions, and a data pipeline that only feeds it measured failures.

## Tool calling is five decisions, not a generation problem

Most tool-calling models are language models that write JSON. Ours is a language
model that makes five decisions, with the JSON assembled around it by a compiler.

Given a query and a catalog of typed functions, something has to decide:
(1) does any tool apply, (2) which one, (3) which optional arguments are
licensed by the query, (4) what values go in them, (5) is another call needed.
Every other token — braces, quotes, commas, and critically *every argument
key* — is determined by the schema before the model runs.

A model that generates JSON as text spends capacity learning that `{` follows
`[`. At 45M parameters you cannot afford that. So a grammar compiled from the
declared schemas force-feeds all structure, and the model is consulted only at
the five choice points. Malformed JSON, invented argument names, and calls to
nonexistent tools become unreachable rather than unlikely: 100% well-formed
output, by construction, against their 93.4%.

Measured honestly, the grammar is a *reliability* mechanism, not an accuracy
one — on Mobile Actions, free generation and constrained decoding agree on 150
of 150 rows. What it buys is that the worst failure modes cannot be expressed.

## The equation that ran the project

Strict exact match factors cleanly: row accuracy = P(name sequence) x p^n,
where p is per-call argument accuracy and n the number of calls. That equation
was the project's management structure. Each cycle: measure both factors,
diagnose the binding one, attack only that, re-measure.

- **v4** (24.3 on Seal): name sequencing 80%, p=0.59. Both weak.
- **v5** (31.4): a rebuilt 16k tokenizer with digits as singletons, 350k new
  corpus rows, and the eval-twin train split upweighted 6x. Name sequencing
  91.5% — solved. p barely moved. 1.2 points short.
- **v6** (33.1): nothing but p, attacked bucket by bucket.

The v5 failure diagnostic said: of 193 failing calls, 66 added exactly one
optional argument the query never mentioned, ~35 bound the right value to the
wrong slot, ~30 missed a canonical date format, and ~29 were unwinnable — the
benchmark's own gold stores numeric-typed arguments as strings 13% of the time,
with 74% of parameters inconsistent about it. (We measured the ceiling of any
typing policy: 87.8%. Needle eats the same noise. We stopped chasing it.)

So the v6 data round was three synthesis modes aimed at three buckets:
schemas rich in optionals where the gold uses only the mentioned ones; queries
carrying two dates or two amounts that must land in different slots; natural
language dates that the call must render as ISO — behind a deterministic
checker so the evidence rule never loosens. $56 of a cheap teacher model,
74,250 validated rows, plus 39k omission exemplars mined for free from the
corpus we already had. A mid-training probe made the causality clean: +3.3
Seal points at constant learning rate, before any decay, attributable to
nothing but the corrective data.

## Anneal, don't retrain — and pick your judge carefully

We trained twins on the new corpus. One from scratch. One resumed from the old
model's plateau checkpoint, with the training data *switched to a
corrective-heavy blend during the learning-rate decay* — the phase where a
WSD-trained model crystallizes. The literature (MiniCPM, Llama 3, OLMo 2)
recommends this; now we have a controlled reading of why.

The scratch twin got the better validation loss — and scored 28.4 on Seal, a
three-point regression. The annealed twin scored 33.1. Same data, same
architecture, same budget. Fed from step zero, corrective data dilutes into
the average; annealed into a trained model during decay, it concentrates
exactly where behavior sets.

Which produced our most instructive failure: the pre-registered champion
selector — held-out loss on the general training mix — picked the scratch
twin, by five thousandths of a nat. A general-mix dev set structurally
penalizes a model that just annealed *away* from the general mix toward the
target distribution. The selector was biased against the winning recipe by
construction. Both models' full tables are published, the selection failure is
documented, and the fix (rank candidates on an eval-flavored but eval-free
split) is one line in the repo for the next cycle.

## The graveyard is the moat

The repo's RESULTS.md records every idea that died, with the measurement that
killed it: span-copy heads (-30), pointer heads (-16), beam/RL search (oracle-
capped below the target), draft-then-constrain (no tax to recover), down-
weighting grammar-forced tokens (-12: structure tokens carry the sequencing
signal), field-set reranking (-1.4: training had already eaten its lunch),
RLOO fine-tuning on the annealed checkpoint (diverges at every learning rate —
sharp minima and policy gradients do not mix). Two of those would have shipped
on intuition. The discipline that nothing ships without an A/B on a suite it
wasn't designed for is, as far as we can tell, the actual moat.

What we could not engineer around: BFCL. Their 42.6 rests on breadth — Java
and JavaScript schema dialects, values the query paraphrases instead of
stating, open-domain coverage — that a 1B-token extractive corpus does not
contain and a 48M model fed concentrate cannot infer. Needle proves 45M
params can hold that number; matching it is a data-volume project (~$500 of
GPU and tens of billions of broad tokens), not a cleverness project. That is
the honest boundary of this result: concentration beats volume exactly where
the test is narrow, and nowhere else.

## Numbers, disclosures, receipts

48.12M params fp32 (11.5MB at their 2-bit standard, vs their 14MB at 45.0M) —
parameter-class-matched, disclosed exactly. Their metric, their tables,
their published numbers. DroidCall carries a caveat (their split is unseeded;
ours is seeded from the same pool, firewalled from training). Every training
row passed an 8-gram contamination firewall against every eval query. The
Seal-in win is +0.5 on 700 rows and is stated as within noise.

One model, one row, no per-suite checkpoint shopping. The full experimental
record — including the twin that dev loss wrongly crowned — is in the repo.

*Built by one person and an AI assistant in about a week of evenings.
Everything is MIT. The failures are the useful part.*

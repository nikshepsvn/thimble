<div align="center">

# 🧵 Thimble

### A tool-calling layer, not a language model.

**Your schemas in, validated calls out, at 48M parameters.**

[![License: MIT](https://img.shields.io/badge/license-MIT-1f6feb?style=flat-square)](LICENSE)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-thimble--v6-ffcc4d?style=flat-square)](https://huggingface.co/flashvenom/thimble)
[![Parameters](https://img.shields.io/badge/params-48.12M-c8324c?style=flat-square)](#numbers)
[![Well-formed JSON](https://img.shields.io/badge/well--formed%20JSON-100%25%20by%20construction-2ea043?style=flat-square)](#the-contract)
[![Build cost](https://img.shields.io/badge/total%20build%20cost-%24260-8957e5?style=flat-square)](REPRODUCING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.png">
  <img alt="Accuracy by suite, split by whether the tool catalog appeared in training" src="assets/results-light.png" width="100%">
</picture>

</div>

It does not converse, reason, or write prose — it was never trained to. It reads
a catalog of typed functions and a request, and returns the calls to make or an
empty list when nothing fits.

That narrowness is the design, not a limitation of it. The tokenizer, the
training loss, and the decoder are all built around the same five decisions, so
the model is never asked to spend capacity on JSON it will never emit. The whole
job then fits in 48M parameters — small enough that specializing it to one API
surface is routine rather than a project.

## The contract

Three guarantees hold on **any** catalog, with no training and no configuration,
because they come from a grammar compiled out of your schemas rather than from
the weights:

- **Output is always well-formed JSON.** Not usually — always. Malformed output
  is unreachable, not unlikely.
- **Argument keys come from your schema.** Parameter-name hallucination is
  structurally impossible.
- **Calls to tools you did not declare cannot be emitted.**

The model is consulted at exactly five choice points: refuse-or-call, which tool,
include this optional, what value, stop or continue. Everything else — braces,
quotes, commas, every argument key — is determined before it runs.

Accuracy is a separate question, answered below with numbers. The contract is not
conditional on any of them.

## Try it in 30 seconds

```
git clone https://github.com/nikshepsvn/thimble && cd thimble
uv venv && uv pip install -e ".[hub]"

# both files come from the HF repo; neither is in git
hf download flashvenom/thimble thimble-v6.pt --local-dir checkpoints/
hf download flashvenom/thimble tokenizer.json --local-dir data/
```

Then:

```
$ python demo.py "make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on"
[
  {"name": "createReservation",
   "arguments": {"partySize": 2, "restaurant": "Nobu", "time": "7pm"}},
  {"name": "sendMessage",
   "arguments": {"body": "dinner is on", "contact": "Sam"}}
]

$ python demo.py "sing me a happy birthday song"
[]  (refused: no tool applies)
```

Real output, not a mock — the typed integer `partySize`, the two-call
composition, and the refusal. Point it at your own tools with:

```
python scripts/eval_catalog.py --ckpt thimble-v6 \
    --catalog my_tools.json --gold my_eval.jsonl
```

## Does this fit your problem?

**It works out of the box when** requests are command-shaped and state their
values: `annotate variant rs4988235 against build GRCh38`. Identifiers, codes,
dates, numbers, enum picks — copied, not inferred. Chains are fine: two-plus-call
rows score **73.5%** on a catalog it knows.

The gate is how *extractive* the request is, not what domain it belongs to. In a
pair of small probes, an unseen biomedical catalog in `dot.notation` scored 0.75
while a familiar-looking app catalog with conversational phrasing scored 0.57.
Unfamiliar vocabulary is survivable; phrasing that hides the values is not.
(Two hand-written probes, 15 rows — directional, not a measurement.)

**Adapt it when** you need conversational phrasing, disciplined handling of
optional arguments, or calibrated refusal. Those three are what specializing
buys, and they are the documented weak spots — see below.

**Use something else when** you have an open-world catalog, need Java or
JavaScript schema dialects, or need parallel instantiations of one schema. And
if you can afford 600M parameters, fine-tune Qwen instead — it will probably
score higher. This is for when you cannot: a memory ceiling, a latency floor, or
wanting a separate model per customer rather than one prompted model for all.

## Numbers

Ordered strict exact match — a row passes only if the function names, the call
order, and every argument value match. The right-hand column is a yardstick, not
a rival: Needle 2 (Cactus Compute, 45M params, 153B training tokens), their
published numbers on their metric, unmodified. It is there so the left column
has a scale.

**Catalog represented in training** (eval rows firewalled out):

| suite | Thimble v6 | Needle 2 (45M) |
|---|---:|---:|
| Mobile Actions (961) | **86.3** | 63.7 |
| Mobile Actions, two-plus-call rows | **73.5** | 48.4 |
| DroidCall (200) | **52.5** | 17.0 |
| Seal-Tools in-domain (700) | **33.1** | 32.6 |
| Well-formed JSON | **100.0** | 93.4 |

**Catalog never seen:**

| suite | Thimble v6 | Needle 2 (45M) |
|---|---:|---:|
| Seal-Tools out-of-domain (654) | 28.1 | 28.7 |
| BFCL v4 single-turn (3,641) | 23.5 | 42.6 |

The spread between those tables is visible *inside a single suite* — the cleanest
control in the project, because only one variable moves:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/catalog-control-dark.png">
  <img alt="Seal-Tools in-domain vs out-of-domain: row accuracy 33.1 vs 28.1, tool-name sequence 88.0 vs 79.0" src="assets/catalog-control-light.png" width="88%">
</picture>

Name-sequence accuracy tracks row accuracy exactly. The model is not failing to
extract arguments on unfamiliar catalogs — it is failing to pick the right
function.

**Disclosures.** Mobile Actions' public train split (8,693 rows, disjoint from
eval) is in the training mix — that is what the first table's heading means.
DroidCall's official split script calls `random.shuffle()` unseeded, so their
exact 200 rows are unreproducible by anyone; ours is a seeded split from the same
pool, firewalled out of training. The Seal-in margin over the yardstick is +0.5
on 700 rows, within sampling noise. The pre-registered dev-loss champion was a
sibling checkpoint scoring 28.4; that selector failure is diagnosed in
[FINDINGS.md](FINDINGS.md) with both models' tables published.

## Adapting it to your catalog

```
python scripts/adapt.py --catalog my_tools.json --name mydomain
python scripts/eval_catalog.py --ckpt mydomain --baseline thimble-v6 \
    --catalog my_tools.json --gold my_eval.jsonl
```

Three stages, each resumable with `--stage`:

| stage | what happens |
|---|---|
| `synth` | a teacher model writes (query → calls) rows **against your schemas**; each is validated against your parameter types and an evidence rule before it is kept |
| `pack` | your rows are blended with guard corpora and packed into two splits |
| `train` | continues from `thimble-v6`, annealing your blend into the LR-decay phase |

Two things about that recipe are load-bearing, both measured rather than assumed:

- **Anneal, don't retrain.** The same corrective corpus scored 28.4 fed from
  scratch and 33.1 annealed into the decay phase. Corrective data dilutes into
  the average when it competes with a whole corpus; it concentrates when it
  arrives late.
- **Keep the guard data.** The blend deliberately carries general tool-calling
  rows alongside yours. Annealing purely on your catalog trades away the
  competence you are building on. `adapt.py` warns if it finds none.

Pass `--examples` if you have real gold rows; they are weighted above synthetic
ones. Needs `OPENROUTER_API_KEY` for synthesis and a GPU to train. For scale, the
v6 cycle synthesized 74,250 validated rows for $56.

**Not yet demonstrated end to end.** `adapt.py` wires together exactly the
machinery that produced the v6 result, but no third-party catalog has been
adapted and published. The recipe is measured; the ergonomics are new. If you run
it, the numbers are worth a pull request.

## How it works

Most constrained-decoding systems bolt a grammar onto a model trained to generate
free text, then manage the mismatch. Here the **tokenizer, the training loss, and
the decoder are one design**, built around the same five decision points:
refuse-or-call, which tool, include this optional, what value, stop or continue.

**The tokenizer is built for the grammar.** JSON structural characters — and
digits — are singleton tokens. Structure can therefore be force-fed *exactly*,
with no token-healing and no ambiguity about where a constraint lands. The usual
arrangement masks logits over a vocabulary that merged `",` into a single token
and papers over the seam. Digits never merge either, so a copied number tokenizes
the same way every time; the rebuild was verified by a fragmentation gate
(word-value fragmentation 2.72 → 2.34 tokens/word, digits lengthening by design).
It shipped as part of the v4 → v5 bundle that took name-sequence accuracy from
80.4% to 91.5% — that bundle also added 350k corpus rows and reweighted the mix,
so the tokenizer's own share of the gain was never isolated.

**The loss is weighted by those same five decisions** — structure 1x, keys 1.5x,
names 2x, values 4x, stop-decision 6x — matched to the measured error
distribution. The model is optimized for the choices it will be asked to make,
not for tokens it will never emit.

**The decoder consults the model only at those points.** Everything else is
determined before it runs, which is where [the contract](#the-contract) comes
from. One call, start to finish — `MODEL` marks the only places the network is
asked anything:

```
  [                                    <- grammar
  └─ ? refuse or call ...................... MODEL
       │
       ├─ refuse ──────────────► ]      <- grammar
       │
       └─ call
          {"name":"                     <- grammar
          └─ ? which tool .................. MODEL
             ","arguments":{            <- grammar
             │
             ├─ next key from YOUR schema  <- grammar
             │  ├─ ? include it ........... MODEL
             │  └─ ? what value ........... MODEL
             │     (repeat for each key)
             │
             }}                         <- grammar
             └─ ? stop or continue ........ MODEL
                ├─ continue ──► back to {"name":"
                └─ stop ──────► ]        <- grammar
```

Every `<- grammar` line is emitted without consulting the model at all. Argument
keys are iterated from your schema, which is why inventing one is not a
low-probability event — there is no step at which it could happen.

### Evidence the co-design works

Two measurements that look like caveats in isolation are the proof in context.

**There is no projection tax.** The same rows decoded with the grammar and with
no grammar at all (`scripts/draft_vs_constrained.py`):

| suite | free generation | grammar-constrained |
|---|---|---|
| Mobile Actions (150) | 78.7 | 78.7 |
| Seal-Tools in (150) | 26.7 | 28.0 |

On Mobile Actions the two agree on **150 of 150 rows**. The grammar is not
overriding the model — the model already wants what the grammar enforces. A
bolted-on grammar produces disagreement and a tax to recover; this is why
draft-then-constrain (DCCD) had nothing to recover here and was abandoned.

Stated plainly, because the distinction matters: the grammar buys *reliability*,
not accuracy. "Constrained decoding makes the model correct" would be a different
claim and not one this data supports. What it buys is that the worst failure
modes cannot be expressed, plus parseability on the ~11% of Seal rows where free
generation emits invalid JSON.

**And the co-design is load-bearing, not decorative.** Down-weighting the
grammar-forced tokens in the loss — on the theory that the model need not learn
what the decoder will supply — cost **12 points** in a controlled twin run. Those
tokens carry the call-sequencing signal: the model learns *when a call ends*
through structure it never has to emit. Remove them and it breaks.

<details>
<summary><b>The rest of the stack — retriever, name head, trunk</b></summary>


- **Retriever** — `retrieve(query, tools, emitted=...)`, a DTDR-style
  (arXiv 2512.17052) refresh conditioned on the *partial plan*, so the candidate
  set is recomputed after each emitted call rather than once per request.
- **Name head** — a bilinear readout scoring candidate tool-name spans in the
  prompt against the hidden state at the decision position. Selection is treated
  as pointing at the prompt, not generating from a vocabulary, following "Looking
  Is Not Picking" (arXiv 2606.16364): mis-selection is a readout failure, not a
  perception one. Its only positive result was on *unfamiliar* catalogs (+2.2),
  which is why it is on by default for your own tools.
- **Trunk** — deep-thin and gated: d=448, 20 layers, GQA 8/4, SwiGLU x2.0,
  QK-norm, sandwich RMSNorm, tied embeddings, Muon on 2D weights and AdamW on
  embeddings, norms and heads. This part is standard modern practice and is not
  where the advantage is; a controlled study from the Needle authors
  (arXiv 2607.18363) finds architecture choices at this scale worth hundredths of
  a nat at matched parameters. The co-design above is the part that matters.

</details>

<details>
<summary><b>How the model was built — the error-driven data loop</b></summary>


Row accuracy factors as `P(name sequence) x p^n`, where `p` is per-call argument
accuracy. Each version measured which factor was binding and attacked only that:

| version | name seq | p | Seal-in | what changed |
|---|---|---|---|---|
| v4 | 80.4% | 0.593 | 24.3 | baseline |
| v5 | 91.5% | 0.60 | 31.4 | 16k digit-singleton tokenizer, +350k corpus rows, seal_train x6, dev-selected EMA |
| **v6** | ~92% | ~0.66 | **33.1** | error-driven synth against three measured failure buckets, annealed into the decay phase |

The v6 data round came straight from the v5 diagnostic: of 193 failing calls, 66
added exactly one unmentioned optional, ~35 bound the wrong entity, ~30 missed
canonical date forms, ~29 were unwinnable noise in the gold. Mid-run causal
check: **+3.3 points at constant LR** from the corrective corpus alone. That loop
is what `adapt.py` automates for your catalog.

</details>

## What did not work

Eleven ideas, each killed by a measurement rather than an argument: span copying
(−30), pointer heads (−16), RFT-style loss down-weighting (−12), from-scratch
retraining (−4.7), field-set reranking (−1.4), beam/RL/best-of-N (oracle-capped
below target), draft-then-constrain (no tax to recover), a global optional-skip
prior (catalog-dependent), `MAX_CALLS` (never binding), RLOO on an annealed
checkpoint (diverges at every LR), and matching the benchmark's numeric typing
(not learnable). Plus two process failures that cost real points.

**[FINDINGS.md](FINDINGS.md) has all of them** with the measurement, the reason,
and the takeaway. It is the most reusable part of the project.

## Known limits

- **Unfamiliar catalogs.** Out-of-domain name-sequence accuracy is 79% against
  88% in-domain. Every out-of-domain deficit traces to this number — the one
  `adapt.py` exists to move.
- **Optional arguments, in both directions.** The largest documented failure
  bucket: 66 of 193 failing v5 calls added exactly one optional the query never
  mentioned, and the model also drops optionals the query does state.
- **Multi-call tracks per-call accuracy, not call count.** `P(names) x p^n`, so
  chains collapse wherever `p` is mediocre and hold where it is not: 73.5% on
  Mobile Actions, 19.4% on Seal-Tools in-domain. The call count is not the
  problem; the catalog is.
- **Parallel calls are a separate, worse failure.** `parallel` 12.0,
  `live_parallel` 0.0 — repeated instantiations of one schema, as opposed to
  calls the query motivates in sequence.
- **Schema dialects.** `simple_python` 29.3 on BFCL against `simple_java` 14.0
  and `simple_javascript` 8.0. Those conventions are absent from a deliberately
  extractive ~1B-token corpus.
- **768-token context.** 151 of 3,641 BFCL rows (4.1%) do not fit and score as misses.
- **Deployment.** 48.12M parameters is ~11.5MB at 2-bit, but 2-bit would need
  quantization-aware retraining this model never had. What actually runs today:
  [cengine/](cengine/) is a dependency-free single-file C port of the full
  decoder — 48MB int8 weights, ~20ms load, byte-identical output to the Python
  stack on 100/100 checked eval rows, faster than the torch stack on an M3.
  That makes laptops, phones and edge Linux reachable; microcontrollers are not.

Scale explains most of it honestly: ~1B unique tokens, no pretraining phase, a
corpus spent deliberately on depth rather than breadth.

## Going deeper

- **[FINDINGS.md](FINDINGS.md)** — eleven negative results and two process
  failures, with the measurement that killed each one. The most reusable part.
- **[REPRODUCING.md](REPRODUCING.md)** — repository layout and the exact
  pipeline that rebuilds the published numbers.
- **[RESULTS.md](RESULTS.md)** — the full chronological experimental record.

## Honest summary

Turning a request into calls against an API you control is a smaller problem than
the models usually pointed at it. Treated as a translation layer rather than a
language model, it fits in 48M parameters, comes with guarantees a prompted model
cannot offer, and can be specialized to one catalog for the price of a dinner.
Its limits are real and measured rather than described. Built by one person over
a few days with AI assistance, for about the price of a video game console.

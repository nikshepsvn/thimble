# 🧵 Thimble

**Tool calling in 48M parameters — small enough to specialize per domain.**

Most tool-calling models are trained once and prompted everywhere. This one is
built to be *re-specialized*: at 48M parameters, adapting it to your own tool
catalog is a few hours on one GPU and roughly $60 of synthesis, so you can have
a model **per** domain instead of a prompt per domain.

Calling tools against a catalog you control is a structured extraction problem,
not a reasoning problem. The repo ships the model, the adaptation loop, the
evidence for why annealing beats retraining, and [the eleven things that did not
work](FINDINGS.md).

**[Model on Hugging Face](https://huggingface.co/flashvenom/thimble)** ·
[Findings](FINDINGS.md) · [Full experimental record](RESULTS.md) · MIT · ~$260 to build

![Results](assets/results.png)

## Try it in 30 seconds

Download `thimble-v6.pt` from the [HF repo](https://huggingface.co/flashvenom/thimble)
into `checkpoints/`, then:

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

Real output, not a mock — note the typed integer `partySize`, the two-call
composition, and the refusal.

## Make it yours

Point it at your schemas and see where you stand:

```
$ python scripts/eval_catalog.py --ckpt thimble-v6 \
      --catalog examples/catalog.json --gold examples/eval.jsonl
7 rows · catalog.json · device cpu
model  thimble-v6  accuracy=0.571  name_acc=0.714  well_formed=1.000  refuse=0.500
```

That is the honest out-of-the-box number on a catalog the model has never seen —
and the failures are the informative part. It drops `label: "gym"` from an alarm
it otherwise gets right, refuses a two-call request it should have answered, and
invents a `getWeather` call for "what is the capital of France". Unfamiliar
catalog, unfamiliar phrasing, exactly as the tables below predict.

Then adapt it:

```
python scripts/adapt.py --catalog my_tools.json --name mydomain
python scripts/eval_catalog.py --ckpt mydomain --baseline thimble-v6 \
    --catalog my_tools.json --gold my_eval.jsonl
```

`adapt.py` runs three stages, each resumable with `--stage`:

| stage | what happens |
|---|---|
| `synth` | a teacher model writes (query → calls) rows **against your schemas**; each row is validated against your parameter types and an evidence rule before it is kept |
| `pack` | your rows are blended with guard corpora and packed into two splits |
| `train` | continues from `thimble-v6`, annealing your blend into the LR-decay phase |

Two things about that recipe are load-bearing, and both were measured rather
than assumed:

- **Anneal, don't retrain.** The same corrective corpus scored 28.4 fed from
  scratch and 33.1 annealed into the decay phase. Corrective data dilutes into
  the average when it competes with a whole corpus and concentrates when it
  arrives late.
- **Keep the guard data.** The blend deliberately carries general tool-calling
  rows alongside yours. Annealing purely on your catalog trades away the general
  competence you are building on. `adapt.py` warns if it cannot find any.

Bring your own gold rows with `--examples` if you have them — real examples are
weighted above synthetic ones, because they are the actual distribution rather
than a teacher's guess at it.

Needs `OPENROUTER_API_KEY` for the synth stage and a GPU for the train stage.
For reference, the v6 cycle synthesized 74,250 validated rows for $56.

**Not yet demonstrated end to end.** `adapt.py` wires together exactly the
machinery that produced the v6 result, but no third-party catalog has been
adapted and published yet. The recipe is measured; the ergonomics are new.
If you run it, the numbers are worth a pull request.

## When to use this

Reach for it when you have a **fixed catalog you control**, English queries,
argument values that appear in the query, and a reason to care about 48M
parameters — a memory ceiling, a latency floor, or wanting a separate model per
customer rather than one prompted model for all of them.

Do not reach for it for open-world tool catalogs, Java or JavaScript schema
dialects, or parallel calls. And if you can afford 600M parameters, fine-tune
Qwen instead — it will probably score higher. This is for when you cannot.

## What it does

Ordered strict exact match: a row passes only if the function names, the call
order, and every argument value match. The right-hand column is a yardstick, not
a rival: Needle 2 (Cactus Compute, 45M params, 153B training tokens), their
published numbers on their metric, unmodified. It is there so the left column
has a scale — 86.3 means little until you know what else scores on that suite.

**Known catalog** — the catalog was represented in training, eval rows firewalled out:

| suite | Thimble v6 | Needle 2 (45M) |
|---|---:|---:|
| Mobile Actions (961) | **86.3** | 63.7 |
| DroidCall (200) | **52.5** | 17.0 |
| Seal-Tools in-domain (700) | **33.1** | 32.6 |
| Well-formed JSON | **100.0** | 93.4 |

**Unknown catalog** — schemas the model has never seen:

| suite | Thimble v6 | Needle 2 (45M) |
|---|---:|---:|
| Seal-Tools out-of-domain (654) | 28.1 | 28.7 |
| BFCL v4 single-turn (3,641) | 23.5 | 42.6 |

Those two tables are why the adaptation loop exists. Familiar catalog, it works.
Unfamiliar catalog, it degrades — measurably, *inside a single suite*:
Seal-Tools in-domain 33.1 versus out-of-domain 28.1 is the same model on the
same metric with only the catalogs changed. Name-sequence accuracy tracks it
exactly, 88% in-domain against 79% out. Getting your catalog into the first
column is the whole job.

**Disclosures.** Mobile Actions' public train split (8,693 rows, disjoint from
eval) is in the training mix — that is what "known catalog" means. DroidCall's
official split script calls `random.shuffle()` unseeded, so their exact 200 rows
cannot be reproduced by anyone; ours is a seeded split from the same pool with
those rows firewalled out of training. The Seal-in margin over the yardstick is
+0.5 on 700 rows, within sampling noise. The pre-registered dev-loss champion
was a sibling checkpoint that scored 28.4; the selector's failure mode is
diagnosed in [FINDINGS.md](FINDINGS.md), with both models' tables published.

## How it works

A deep-thin gated trunk (d=448, 20 layers, GQA 8/4, SwiGLU x2.0, QK-norm,
sandwich RMSNorm, tied embeddings) plus three factorized capabilities:

1. **Retriever** — `retrieve(query, tools, emitted=...)`, a DTDR-style
   (arXiv 2512.17052) refresh conditioned on the *partial plan*. These benchmarks
   are single-pass with no tool results, so the refresh happens after each
   emitted call.
2. **Name head** — a bilinear readout scoring candidate tool-name spans in the
   prompt against the hidden state at the decision position, motivated by
   "Looking Is Not Picking" (arXiv 2606.16364): tool mis-selection is a readout
   failure, not a perception one. Its only positive result was on *unfamiliar*
   catalogs (+2.2), which is why it is on by default for your own tools.
3. **Grammar decoder** — the tokenizer keeps JSON structural characters as
   singleton tokens, so structure is force-fed exactly and the model is consulted
   at only five choice points: refuse-vs-call, name, optional-include, value
   content, stop-vs-continue. **Argument keys are forced from the schema**, so
   parameter-name hallucination is structurally unreachable — on your catalog as
   much as on ours, with no training required.

Training is weighted cross-entropy matched to the observed error distribution
(structure 1x, keys 1.5x, names 2x, values 4x, stop decision 6x) plus a name-head
auxiliary loss, with Muon on trunk 2D weights and AdamW on embeddings, norms and heads.

### What the grammar actually buys

Measured, not assumed. Decoding the same rows with the grammar and with no
grammar at all (`scripts/draft_vs_constrained.py`):

| suite | free generation | grammar-constrained |
|---|---|---|
| Mobile Actions (150) | 78.7 | 78.7 |
| Seal-Tools in (150) | 26.7 | 28.0 |

On Mobile Actions the two agree on **150 of 150 rows**. The grammar is not an
accuracy mechanism here — it is a *reliability* mechanism. It buys unconditional
well-formedness, unreachable parameter-name hallucination, and parseability on
the ~11% of Seal rows where free generation emits invalid JSON. That is worth
having, and it is not the same claim as "constrained decoding makes the model
correct".

## The recipe

Row accuracy factors as `P(name sequence) x p^n`, where `p` is per-call argument
accuracy. Every version was built by measuring which factor was binding and
attacking only that:

| version | name seq | p | Seal-in | what changed |
|---|---|---|---|---|
| v4 | 80.4% | 0.593 | 24.3 | baseline |
| v5 | 91.5% | 0.60 | 31.4 | 16k digit-singleton tokenizer, +350k corpus rows, seal_train x6, dev-selected EMA |
| **v6** | ~92% | ~0.66 | **33.1** | error-driven synth aimed at three measured failure buckets, annealed into the decay phase |

The v6 data round was designed directly from the v5 failure diagnostic: of 193
failing calls, 66 added exactly one unmentioned optional, ~35 bound the wrong
entity, ~30 missed canonical date forms, and ~29 were unwinnable type-convention
noise in the gold itself. Mid-run causal check: **+3.3 Seal points at constant
LR**, attributable purely to the corrective corpus.

That loop — diagnose the buckets, synthesize against them, anneal them late — is
what `adapt.py` automates for your catalog.

## What did not work

Eleven ideas, each killed by a measurement rather than an argument: span
copying (−30), pointer heads (−16), RFT-style loss down-weighting (−12),
from-scratch retraining (−4.7), field-set reranking (−1.4), beam/RL/best-of-N
(oracle-capped below target), draft-then-constrain (no tax to recover), a global
optional-skip prior (catalog-dependent), `MAX_CALLS` (never binding), RLOO on an
annealed checkpoint (diverges at every LR), and matching the benchmark's numeric
typing (not learnable). Plus two process failures that cost real points.

**[FINDINGS.md](FINDINGS.md) has all of them** with the measurement, the reason,
and the takeaway. It is the most reusable part of the project.

## Where it stops working

- **Unfamiliar catalogs.** Out-of-domain name-sequence accuracy is 79% against
  88% in-domain. Every out-of-domain deficit traces back to this one number —
  and it is the one `adapt.py` exists to move.
- **Schema dialects.** `simple_python` scores 29.3 on BFCL, but `simple_java`
  14.0 and `simple_javascript` 8.0. Java and JS conventions are absent from a
  deliberately extractive ~1B-token corpus.
- **Parallel calls.** `parallel` 12.0 and `live_parallel` 0.0. Multi-call
  composition works when calls are sequentially motivated by the query, not when
  they are parallel instantiations of one schema.
- **768-token context.** 151 of 3,641 BFCL rows (4.1%) do not fit and score as misses.
- **Deployment.** 48.12M parameters is ~11.5MB at 2-bit and ~92MB at bf16, but
  what ships here is the 184MB fp32 checkpoint and **there is no on-device
  inference engine**. The size figure is a property of the parameter count, not
  of a runnable microcontroller artifact. Today this is a small, fast
  server-side model.

Scale is the honest explanation for most of this: ~1B unique tokens, no
pretraining phase, a corpus deliberately spent on depth rather than breadth.

## Layout

```
src/tiny_toolcall/
  model.py       trunk + name head + pointer head (pointer disabled, see FINDINGS)
  grammar.py     constrained decoder; the five choice points
  retrieve.py    DTDR retriever + lexical prior
  render.py      compact tool signatures + per-token loss tags
  official.py    Mobile Actions / Seal-Tools adapters
  bfcl.py        BFCL v4 adapter + AST-equivalent scorer
  teacher.py     synthesis: invented catalogs, and your catalog (synth_for_catalog)
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
held-out dev loss (`scripts/select_champion.py`) — and see FINDINGS.md for why
that selector must be eval-distribution-flavored when recipes anneal.

Requires `.env` with `OPENROUTER_API_KEY` (synthesis) and `RUNPOD_API_KEY` (GPU),
both gitignored. Total cost across all three cycles: about $260; the v6 cycle
alone, which produced the headline numbers, about $85.

## Honest summary

Tool calling against a catalog you control is a smaller problem than the models
usually deployed for it. At 48M parameters, with no pretraining phase and ~1B
unique tokens, this one reaches 86.3% strict exact match on app intents and
cannot emit malformed JSON or hallucinate a parameter name. Off a familiar
catalog it degrades sharply — which is why the interesting artifact is not the
checkpoint but the loop that re-specializes it, and the record of what failed
along the way. Built by one person over a few days with AI assistance, for about
the price of a video game console.

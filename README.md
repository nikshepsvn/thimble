# 🧵 Thimble

**Tool calling in 48M parameters.** 86.3% ordered strict exact match on a real
app-intent catalog, 100% well-formed JSON by construction, MIT.

Calling tools against a *known* catalog is not an emergent capability of large
models — it is a structured extraction problem, and it fits in 48M parameters.
This repo is the model, the recipe that produced it, and the measured point where
the approach stops working.

**[Model on Hugging Face](https://huggingface.co/flashvenom/thimble)** ·
[The full experimental record](RESULTS.md) · Total build cost: ~$260

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
composition, and the refusal. `python demo.py` with no arguments runs a small
example set.

## What it does

Ordered strict exact match: a row passes only if the function names, the call
order, and every argument value match. The right-hand column is calibration —
Needle 2 (Cactus Compute, 45M params, 153B training tokens), their published
numbers, their metric, unmodified — so you can judge whether these numbers are
any good. It is a reference point, not the thesis.

**Known catalog** — the catalog was represented in training, eval rows firewalled out:

| suite | Thimble v6 | for calibration |
|---|---:|---:|
| Mobile Actions (961) | **86.3** | 63.7 |
| DroidCall (200) | **52.5** | 17.0 |
| Seal-Tools in-domain (700) | **33.1** | 32.6 |
| Well-formed JSON | **100.0** | 93.4 |

**Unknown catalog** — schemas the model has never seen:

| suite | Thimble v6 | for calibration |
|---|---:|---:|
| Seal-Tools out-of-domain (654) | 28.1 | 28.7 |
| BFCL v4 single-turn (3,641) | 23.5 | 42.6 |

Those two tables are the entire finding. Familiar catalog, it works. Unfamiliar
catalog, it degrades — and the degradation is measurable inside a *single* suite:
Seal-Tools in-domain 33.1 versus out-of-domain 28.1 is the same model on the same
metric, with only the catalogs changed. Name-sequence accuracy tracks it exactly,
88% in-domain against 79% out.

**Disclosures.** Mobile Actions' public train split (8,693 rows, disjoint from
eval) is in the training mix; that is what "known catalog" means, and it is the
intended operating condition rather than a caveat. DroidCall's official split
script calls `random.shuffle()` unseeded, so their exact 200 rows cannot be
reproduced by anyone — ours is a seeded split from the same pool with those rows
firewalled out of training. The Seal-in margin over the calibration column is
+0.5 on 700 rows, which is within sampling noise and should not be read as a win.
The pre-registered dev-loss champion was a sibling checkpoint that scored 28.4 on
Seal-in; the model above is the annealed recipe's within-run dev winner. The
selector's failure mode — a general-mix dev cannot rank a decay-annealed model —
is diagnosed in [RESULTS.md](RESULTS.md), with both models' full tables published.

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

## The recipe

Row accuracy factors as `P(name sequence) x p^n`, where `p` is per-call argument
accuracy. Every version was built by measuring which factor was binding and
attacking only that:

| version | name seq | p | Seal-in | what changed |
|---|---|---|---|---|
| v4 | 80.4% | 0.593 | 24.3 | baseline |
| v5 | 91.5% | 0.60 | 31.4 | 16k digit-singleton tokenizer, +350k corpus rows, seal_train x6, dev-selected EMA |
| **v6** | ~92% | ~0.66 | **33.1** | error-driven synth aimed at the three measured failure buckets, annealed into the decay phase |

The v6 data round was designed directly from the v5 failure diagnostic: of 193
failing calls, 66 added exactly one unmentioned optional (fixed by omission-pressure
synth + 39k mined exemplars), ~35 bound the wrong entity (entity-distractor synth),
~30 missed canonical date forms (ISO-date synth behind a deterministic evidence
checker), and ~29 were unwinnable type-convention noise in the gold itself.
Mid-run causal check: +3.3 Seal points at constant LR, attributable purely to the
corrective corpus.

**Where corrective data belongs.** The same corpus, fed two ways, in a controlled
twin run:

| | Seal-in |
|---|---:|
| trained from scratch on the corrective mix | 28.4 |
| annealed into the LR-decay phase of a continued run | **33.1** |

From scratch, corrective data dilutes into the average. Annealed into an already
trained model during decay — where behavior crystallizes — it concentrates. This is
probably the most portable result in the repo.

## What did not work

Eleven plausible ideas, each killed by a measurement rather than an argument. This
is the part most worth reading.

| idea | result | why |
|---|---|---|
| Word-span copying | **-30 pts** | span boundaries are wrong more often than free generation |
| Pointer/copy head (endpoints) | **-16 pts** | head learns, but its confidence does not correlate with correctness |
| `MAX_CALLS` 4 -> 6 | **0.000** | the cap was never binding; long rows fail for other reasons |
| Global optional-skip prior | rejected | inclusion rate is catalog-dependent (33% Seal, 94% Mobile Actions) — a constant trades one suite for another |
| Search: beam / RL / best-of-N | **capped at +5.0** | pass@9 is an *oracle* bound and only reaches 29.3 vs the 32.6 it needed |
| Draft-then-constrain (DCCD) | **no effect possible** | requires a projection tax; measured at -1.3 and +0.0, so there is nothing to recover |
| Down-weighting grammar-forced tokens (RFT-style) | **-12 pts** | controlled twin run; structure tokens carry the call-sequencing signal |
| Field-set reranking (PGR-style) | **-1.4** | by the time it ran, training had already fixed the key-set bucket it targets (fixed 3, broke 13) |
| MRT/RLOO fine-tune | mixed, then unusable | +1.9 out-domain / -1.0 in-domain on v5; diverges outright on the annealed v6 checkpoint at every LR tried |
| Emitting numerics to match Seal's gold typing | **not learnable** | their string-vs-number choice is 74% mixed per parameter; per-param policy ceiling 87.8% vs 86.4% global — noise |
| From-scratch retrain on the corrective corpus | **-4.7 vs annealing** | corrective data dilutes into the average from scratch; annealed into a trained model it concentrates |

Two of these reversed conclusions that would otherwise have shipped on intuition. The
pointer-head result is independently corroborated: pointer generators are reported to
*hurt* structured extraction under grammar-constrained decoding in low-resource settings.

A twelfth belongs here even though it is a process failure rather than an idea: the
pre-registered champion selector picked the wrong checkpoint, and the post-mortem is
in RESULTS.md.

## Where it stops working

The boundary is the point of the two tables above, so it is worth stating precisely
rather than burying.

- **Unfamiliar catalogs.** Out-of-domain name-sequence accuracy is 79% against 88%
  in-domain. Every out-of-domain deficit traces back to this one number.
- **Schema dialects.** `simple_python` scores 29.3, but `simple_java` 14.0 and
  `simple_javascript` 8.0. Java and JS schema conventions are simply absent from a
  deliberately extractive ~1B-token corpus.
- **Parallel calls.** `parallel` 12.0 and `live_parallel` 0.0 on BFCL. Multi-call
  composition works when the calls are sequentially motivated by the query and not
  when they are parallel instantiations of one schema.
- **768-token context.** 151 of 3,641 BFCL rows (4.1%) do not fit and are scored as
  misses.
- **Deployment.** 48.12M parameters is ~11.5MB at 2-bit and ~92MB at bf16, but what
  ships here is the 184MB fp32 checkpoint and there is no on-device inference engine.
  The size claim is a property of the parameter count, not of an artifact you can run
  on a microcontroller today.
- (Fixed since v4: small-catalog refusal — BFCL irrelevance went 0.0 -> 57.9 after
  the refuse-gate fix and small-catalog refusal data.)

Scale is the honest explanation for most of this. The model saw ~1B unique tokens
with no pretraining phase. Breadth is what more data buys, and this corpus was
deliberately spent on depth instead. That trade is priced in RESULTS.md rather than
excused.

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
python scripts/convert_v5.py && python scripts/convert_v5b.py
python scripts/firewall2.py           # 8-gram contamination sweep
python -m tiny_toolcall.cli pack --seq-len 768
python scripts/pack_anneal.py         # decay-phase corrective blend
python -m tiny_toolcall.cli train --name v6c --epochs 2 \
    --init <plateau-checkpoint> --no-warmup --anneal-data anneal
python scripts/final_eval.py --ckpt v6c_ema --suite seal-tools-in
```

`--seq-len 768` is not optional: the 512 default silently drops the longest rows,
which are exactly the multi-call ones. Champion selection is by held-out dev loss
(`scripts/select_champion.py`) — and see RESULTS.md for why that selector must be
eval-distribution-flavored when recipes anneal.

Requires `.env` with `OPENROUTER_API_KEY` (synthesis) and `RUNPOD_API_KEY` (GPU) —
both gitignored. Total cost across all three cycles (v4 through v6): about $260;
the v6 cycle alone, which produced the headline numbers, about $85.

## Honest summary

Tool calling against a catalog you control is a smaller problem than the models
usually deployed for it. At 48M parameters, with no pretraining phase and ~1B unique
tokens, this one reaches 86.3% exact match on app intents and cannot emit malformed
JSON or hallucinate a parameter name. Off that catalog it degrades sharply, and the
degradation is measured rather than described — schema dialects, parallel calls, and
unfamiliar name selection, each quantified. Built by one person over a few days with
AI assistance, for about the price of a video game console.

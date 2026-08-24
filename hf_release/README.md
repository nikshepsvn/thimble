---
license: mit
language:
- en
pipeline_tag: text-generation
tags:
- function-calling
- tool-calling
- on-device
- small-model
- grammar-constrained-decoding
- edge
library_name: pytorch
model-index:
- name: thimble-v6
  results:
  - task:
      type: text-generation
      name: Function calling (ordered strict exact match)
    dataset:
      name: Seal-Tools in-domain
      type: seal-tools
    metrics:
    - type: exact_match
      value: 33.1
      name: Seal-Tools in-domain
  - task:
      type: text-generation
      name: Function calling (ordered strict exact match)
    dataset:
      name: Seal-Tools out-of-domain
      type: seal-tools
    metrics:
    - type: exact_match
      value: 28.1
      name: Seal-Tools out-of-domain
  - task:
      type: text-generation
      name: Function calling (ordered strict exact match)
    dataset:
      name: Mobile Actions
      type: mobile-actions
    metrics:
    - type: exact_match
      value: 86.3
      name: Mobile Actions
  - task:
      type: text-generation
      name: Function calling (ordered strict exact match)
    dataset:
      name: DroidCall
      type: droidcall
    metrics:
    - type: exact_match
      value: 52.5
      name: DroidCall
  - task:
      type: text-generation
      name: Function calling (ordered strict exact match)
    dataset:
      name: BFCL v4 single-turn
      type: bfcl
    metrics:
    - type: exact_match
      value: 23.5
      name: BFCL v4 single-turn
---
# 🧵 Thimble

**Tool calling in 48M parameters.** 86.3% ordered strict exact match on a real
app-intent catalog, 100% well-formed JSON by construction.

[**GitHub (code, evals, full experimental record)**](https://github.com/nikshepsvn/thimble) · MIT · 48.12M params · 768-token context · $260 total build cost

![Results](results.png)

Calling tools against a *known* catalog is not an emergent capability of large
models — it is a structured extraction problem, and it fits in 48M parameters.
This card covers what the model does, how it was built, and exactly where it
stops working.

## What it does

Ordered strict exact match: a row passes only if the function names, the call
order, and *every* argument value match. The right-hand column is a yardstick, not a
rival: Needle 2 (Cactus Compute, 45M params, 153B training tokens), their
published numbers on their metric. It is there so the left column has a scale —
86.3 means little until you know what else scores on that suite.

**Known catalog** — represented in training, eval rows firewalled out:

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

Those two tables are the whole finding. Familiar catalog, it works; unfamiliar
catalog, it degrades — and the degradation shows up *inside a single suite*:
Seal-Tools in-domain 33.1 vs out-of-domain 28.1 is the same model on the same
metric with only the catalogs changed. Name-sequence accuracy tracks it exactly,
88% in-domain against 79% out.

**Before quoting the table.** Mobile Actions' public train split (8,693 rows,
disjoint from eval) is in the training mix — that is what "known catalog" means,
and it is the intended operating condition. The Seal-in margin over the
calibration column is +0.5 on 700 rows, within sampling noise. The pre-registered
model selector picked a sibling checkpoint that scored worse; the failure is
diagnosed and both models' results are published in
[RESULTS.md](https://github.com/nikshepsvn/thimble/blob/master/RESULTS.md).

## How it was built

**1. Tool calling is five decisions, not a generation problem.** A grammar
compiled from the tool schemas force-feeds all JSON structure — braces, quotes,
and every argument key. The model is consulted at exactly five choice points:
*refuse or call · which tool · include this optional? · what value · stop or
continue*. Malformed JSON, hallucinated parameter names, and calls to
nonexistent tools are **unreachable, not unlikely**. At 48M parameters, capacity
spent learning that `{` follows `[` is capacity wasted.

Measured honestly, the grammar is a *reliability* mechanism rather than an
accuracy one — on Mobile Actions, free generation and constrained decoding agree
on 150 of 150 rows. What it buys is that the worst failure modes cannot be
expressed at all.

**2. Every training example earns its place.** Row accuracy factors as
`P(name sequence) × pⁿ`. Each version measured which factor was binding and
attacked only that. The final data round was synthesized directly against the
previous model's diagnosed failure buckets — spurious optional arguments,
wrong-slot entity binding, date canonicalization — with a mid-training causal
check (+3.3 points at constant LR, attributable to the corrective data alone).

**3. Anneal, don't retrain.** A controlled twin experiment: the corrective
corpus fed from scratch *diluted* (28.4); the same corpus **annealed into the
learning-rate decay phase** of a continued run *concentrated* (33.1). The decay
phase is where a WSD-trained model crystallizes — that is where the good data
belongs. This is probably the most portable result in the project.

## What didn't work (measured, not guessed)

The most reusable part of the project. Each idea was killed by an A/B, not an argument:

| idea | result |
|---|---|
| Span-copy heads | −30 pts |
| Pointer/copy head | −16 pts |
| Down-weighting grammar-forced tokens (RFT-style) | −12 pts — structure tokens carry call-sequencing signal |
| From-scratch retrain on corrective data | −4.7 vs annealing |
| Field-set reranking | −1.4 — training had already fixed its target bucket |
| Beam / RL / best-of-N | oracle-capped below target |
| RLOO fine-tune on the annealed checkpoint | diverges at every LR — sharp minima and policy gradients don't mix |
| Matching Seal's gold numeric typing | not learnable — 74% of params are mixed-convention noise |

Two of these reversed conclusions that would otherwise have shipped on intuition.

## Where it stops working

- **Unfamiliar catalogs.** Out-of-domain name-sequence accuracy is 79% against
  88% in-domain. Every out-of-domain deficit traces back to this one number.
- **Schema dialects.** `simple_python` scores 29.3 on BFCL, but `simple_java`
  14.0 and `simple_javascript` 8.0 — Java and JS schema conventions are absent
  from a deliberately extractive ~1B-token corpus.
- **Parallel calls.** `parallel` 12.0 and `live_parallel` 0.0. Multi-call
  composition works when calls are sequentially motivated by the query, not when
  they are parallel instantiations of one schema.
- **768-token context.** 151 of 3,641 BFCL rows (4.1%) do not fit and score as misses.
- **Deployment.** 48.12M parameters is ~11.5MB at 2-bit and ~92MB at bf16, but
  what ships here is the 184MB fp32 checkpoint and there is no on-device
  inference engine. The size figure is a property of the parameter count, not of
  a runnable microcontroller artifact.

Scale is the honest explanation for most of this: ~1B unique tokens, no
pretraining phase, a corpus deliberately spent on depth instead of breadth.

## Model details

| | |
|---|---|
| Parameters | 48.12M (fp32 checkpoint; ~11.5MB at 2-bit) |
| Architecture | deep-thin gated trunk: d=448, 20 layers, GQA 8/4, SwiGLU ×2.0, QK-norm, sandwich RMSNorm, tied embeddings |
| Tokenizer | 16,384 BPE, digits as singletons, JSON structural chars as singletons |
| Context | 768 tokens |
| Decoding | grammar-constrained, five choice points, plan-conditioned retrieval between calls |
| Training | Muon (trunk) + AdamW, WSD schedule, EMA, weighted CE matched to the error distribution, decay-phase data annealing |

## Files & usage

- `thimble-v6.pt` — checkpoint (`torch.load(..., weights_only=False)` → `{"model": state_dict, "cfg": dict}`)
- `tokenizer.json` — BPE vocab + merges

The guarantees live in the decoding harness, so inference goes through the repo:

```bash
git clone https://github.com/nikshepsvn/thimble
cd thimble && uv venv && uv pip install -e .
# put thimble-v6.pt in checkpoints/, tokenizer.json in data/

python demo.py "make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on"
# [{"name": "createReservation",
#   "arguments": {"partySize": 2, "restaurant": "Nobu", "time": "7pm"}},
#  {"name": "sendMessage",
#   "arguments": {"body": "dinner is on", "contact": "Sam"}}]

python demo.py "sing me a happy birthday song"
# []  (refused: no tool applies)

python scripts/final_eval.py --ckpt thimble-v6 --suite seal-tools-in  # reproduce the table
```

Real output, not a mock — typed integers, two-call composition, and refusal,
with structure guaranteed by the grammar.

## Integrity

Public corpora (xlam, ToolACE, Dolci, Glaive, official benchmark train splits)
plus stepwise-validated, evidence-filtered synthetic data. Every training row
passed an **8-gram contamination firewall against every evaluation query of
every reported suite** (BFCL included). Champion selection by held-out dev loss
only; nothing was ever tuned on an eval set; every negative result is published.

*Built by one person and an AI assistant in about a week of evenings, for about
the price of a game console. The failures are the useful part.*

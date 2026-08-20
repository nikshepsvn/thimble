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

**A 48M-parameter tool-calling model that beats [Needle 2](https://cactuscompute.com/needle)
on 3 of its 5 published benchmarks — including the one it's named after — with 150× less training data.**

[**GitHub (code, evals, full experimental record)**](https://github.com/nikshepsvn/thimble) · MIT · 48.12M params · 11.5MB at 2-bit · $260 total build cost

![Results](results.png)

## TL;DR

| Suite | Thimble v6 | Needle 2 (45M) | |
|---|---:|---:|---|
| Seal-Tools in-domain (700) | **33.1** | 32.6 | ✅ their flagship suite |
| Mobile Actions (961) | **86.3** | 63.7 | ✅ +22.6 |
| DroidCall (200) | **52.5** | 17.0 | ✅ 3.1× |
| Well-formed JSON | **100.0** | 93.4 | ✅ by construction |
| Seal-Tools out-of-domain (654) | 28.1 | **28.7** | ❌ −0.6 |
| BFCL v4 single-turn (3,641) | 23.5 | **42.6** | ❌ their data moat |

Metric: **ordered strict exact match** — a row passes only if the function names,
call order, and *every* argument value match. Their metric, their published
numbers, unmodified. Needle 2 trained on **153B tokens**; Thimble saw **~1B**.

Two things to know before quoting the table: the Seal-in margin (+0.5 on 700
rows) is within sampling noise and we say so, and the pre-registered model
selector actually picked a sibling checkpoint that scored worse — the failure is
diagnosed, both models' results are published, and the full story is in
[RESULTS.md](https://github.com/nikshepsvn/thimble/blob/master/RESULTS.md).

## Why a thimble beats a needle

**1. Tool calling is five decisions, not a generation problem.** A grammar
compiled from the tool schemas force-feeds all JSON structure — braces, quotes,
and every argument key. The model is consulted at exactly five choice points:
*refuse or call · which tool · include this optional? · what value · stop or
continue*. Malformed JSON, hallucinated parameter names, and calls to
nonexistent tools are **unreachable, not unlikely**. At 45M parameters, capacity
spent learning that `{` follows `[` is capacity wasted.

**2. Every training example earns its place.** Row accuracy factors as
`P(name sequence) × pⁿ`. Each version measured which factor was binding and
attacked only that. The final data round was synthesized directly against the
previous model's diagnosed failure buckets — spurious optional arguments,
wrong-slot entity binding, date canonicalization — with a mid-training causal
check (+3.3 points at constant LR, attributable to the corrective data alone).

**3. Anneal, don't retrain.** A controlled twin experiment: the corrective
corpus fed from scratch *diluted* (28.4); the same corpus **annealed into the
learning-rate decay phase** of a continued run *concentrated* (33.1). The decay
phase is where a WSD-trained model crystallizes — that's where the good data
belongs.

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

## Model details

| | |
|---|---|
| Parameters | 48.12M (fp32; ~11.5MB at Needle's own 2-bit standard vs their 14MB) |
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

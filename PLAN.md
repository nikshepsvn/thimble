> **Historical planning document.** Written during the v4 cycle and kept as a record
> of what was planned, not what shipped. Parameter counts and targets here are v4-era
> (44M); current numbers are in [README.md](README.md) and [FINDINGS.md](FINDINGS.md).

# v5 — the strongest 44M we can build

Objective: train the strongest tool-calling model this project can produce and
beat Needle 2 on a majority of its published suites. We already hold Mobile
Actions (81.5 vs 63.7) and DroidCall (47.5 vs 17.0); the contested suite is
Seal-Tools in-domain (24.3 vs 32.6). BFCL (15.1 vs 42.6) is out of reach this
cycle and is reported cold as the generalization check, not chased.

**Honest odds, stated up front: this plan is not a 99% beat.** Best estimate
for clearing Seal-in 32.6: **55–65%**. Every input to that estimate is a
measurement from this project, listed below. Anyone promising 99% from here is
selling something.

---

## 1. The one equation that governs the gap

Row accuracy = P(name sequence) x p^n, where p is per-call argument accuracy.

Measured on v4 (200-row shuffled sample, seed 7):
- p = 0.538–0.593, name sequence 83.0%
- beating 32.6 needs p ≈ 0.687
- per-call error buckets: **key-set 28.4%** (65% of it exactly one spurious
  optional), values 17.8%, correct 53.8%
- name selection is NOT the constraint: forcing gold names buys +3 points total

Every intervention below is judged by whether it plausibly moves p or the
key-set bucket. Nothing else matters for Seal.

## 2. The binding constraint is data, not capacity or architecture

- 140M unique training tokens / 44.45M params = **3.1 tokens per param**;
  compute-optimal is ~20. We are 6.4x under-trained at the size we already are.
  Training loss 0.073 confirms the model fits its corpus.
- Therefore: **params stay at 44.45M**. Scaling up worsens the ratio.
- Needle 2 trained on 153B tokens (449x ours) — the gap is a data gap.

## 3. Corpus plan: found data first, synth to fill gaps

### 3a. Import (free, ~890k raw rows)
| source | rows | why |
|---|---|---|
| BitAgent/tool_calling | 551,285 | largest clean single-turn source, explicit schemas |
| allenai/Dolci-Instruct-SFT-Tool-Use | 227,579 | **already BFCL-decontaminated by AllenAI** |
| argilla/apigen-function-calling | 109,402 | format-identical to ours; overlaps xLAM — dedup decides |
| dria step_by_step | 15,884 | re-include; previously excluded by an over-broad filter; 56% multi-call |

Rejected: Nemotron-RL (multi-turn agentic trajectories, wrong shape),
samuki-hf 8.1M (aggregation → duplicates + contamination), glaive reformats
(Locutusque = MathAndMagic = hypervariance = glaive-v2).

### 3b. Cleaning pipeline (order matters, all mandatory)
1. Convert to {query, tools, answers}; first-turn-only for conversational shapes
2. **Evidence filter**: every scalar argument value must appear in the query
   (the rule teacher.py enforces; rejected 18% of dria — wrong-value rows teach
   the exact failure we are fixing)
3. **Dedup** by normalized query across ALL sources (known heavy overlap)
4. **Contamination firewall, hardened**: exact normalized-query match is no
   longer sufficient at ~1M imported rows. Add character n-gram overlap
   (e.g. any 8-gram shared with an eval query → drop) against all 2,515 eval
   queries + BFCL queries. This step is non-negotiable: a contaminated win is
   worthless, and this project's entire value is that its numbers are honest.

Expected net yield: 500k–900k clean rows → 0.3–0.45B unique tokens →
**8–13 tokens/param**, a 3–4x improvement in the binding constraint. Free.

### 3c. Synth top-up (~$20–50, optional, after measuring found-data yield)
The retargeted generator (already committed) fills what found data cannot:
- small-catalog refusals (1–2 tools): zero coverage anywhere; BFCL irrelevance
  is 0.0% because of it
- Seal-dialect rows: camelCase enterprise catalogs of exactly 5 tools,
  chain mix [12,14,38,16,4,16] matched to Seal's real length distribution
  (a dry run at heavier chains produced padded calls — do not push past this)
- numeric-typed args stated as digits (35% of rows), exploiting the type fix

### 3d. Mix corrections
- dria (simple/parallel/multiple) down-weighted: 1.5 tools/row is the prime
  suspect for v4's Mobile Actions 2-call regression (68.8 → 65.1)
- Seal/MA/DroidCall train splits keep their x3; DroidCall stays the
  200-row-heldout file; all 2,515 firewall queries retained

## 4. Tokenizer rebuild (forces the from-scratch run, and is worth it)

- **Vocab 8,192 → 16,384.** Measured fragmentation 3.05–3.69 tokens/word;
  'Golden Gate Towers' = 10 tokens; every fragment is a chance to derail.
  Efficiency plateaus ~32k, so 16k captures most of the gain.
- **Digits are singletons — never merged with anything.** Current segmentation
  is incoherent ('38.0'→['3','8.','0'], '20.8'→['20','.','8'], '2021'→['2021'])
  and produced measured errors (300→'30000', 2.3522→'21.3522'). 77.3% of
  Seal rows and 70.8% of MA rows carry a digit in gold. Literature: base-10 is
  consistently more data-efficient from scratch and the advantage does not
  shrink with scale.
- **No template leakage**: train the tokenizer on queries/values/descriptions,
  not on rendered prompt scaffolding ('(speed=enum(0.5' is currently a vocab
  entry).
- **Keep**: whitespace as its own token (verified: values tokenize identically
  in prompt and target context, 10/10 — do not adopt leading-whitespace merging),
  JSON structural chars as singletons (the grammar contract).

## 5. Model and training

- Architecture: **unchanged** — d=448, 20L, GQA 8/4, SwiGLU x2.0, QK-norm,
  sandwich RMSNorm, gated SDPA, tied embeddings. Cactus's own controlled study
  (arXiv 2607.18363) shows FFN removal is neutral at matched params, and we
  independently match every load-bearing item of their recipe (QK-norm
  mandatory, sandwich norm, 20-layer optimum, Muon, identical loss weights).
  d_model/head count unchanged; the vocab growth adds ~7.3M embedding params —
  acceptable, embeddings are tied.
- Optimizer: Muon (trunk 2D) + AdamW (embeddings/norms/heads), WSD schedule,
  token-budget batching — all unchanged; the SFT literature says our settings
  are already in the good region, and churning them burns the run's attribution.
- Loss: weighted CE (names 2x, values 4x, stop 6x) + name-head aux, PLUS one
  new change: **loss weight 0 on grammar-forced positions** (structure AND
  argument keys — the decoder force-feeds both, so CE spent predicting them is
  capacity spent on determined tokens). Loss masking is established (Llama 2
  prompt masking); applying it to grammar-forced positions appears novel.
  Gate: A/B on a 1-epoch pair if time allows; else ship with the flag exposed.
- Epochs: 3 (v3→v4 precedent), checkpoint every 500 steps. NEW: hold out a
  5k-row dev split from the TRAINING mix and select the best checkpoint on dev
  loss instead of taking the final step. (Dev is train-derived — never eval.)
- **From scratch, not continued**: the tokenizer change forces it; our own
  A/B (v3-scratch 82.6 vs v3-warm 82.3, "the warm start was anchoring")
  independently prefers it.
- Pack with `--seq-len 640` (the 512 default silently drops the longest 12% —
  exactly the multi-call rows).
- Precision: full precision. 2-bit QAT is a deployment story, not a strength
  story — at <50M, low-bit needs ~2x hidden size to break even
  (BitNet-Reloaded), and Needle's numbers being 2-bit already flatters us.
  Revisit only for a byte-parity release, using ParetoQ's 90% FP / 10% QAT.

## 6. Decode-time additions (each gated by its own measurement)

- **Refuse-gate fix**: committed. One-tool catalogs can now refuse (the
  normalized-peak override was structurally unable to).
- **Field-set reranking (PGR-style, arXiv 2608.03071)**: sample k decodes,
  choose the modal KEY SET per call, keep greedy values. Directly targets the
  28.4% key-set bucket, of which 65% is one spurious optional. NOT bounded by
  our pass@k result — pass@k measured selection of whole samples (+5.0 cap);
  this composes. GATE: the field-oracle measurement (running) must show
  composition ceiling meaningfully above the selection ceiling; then the built
  reranker must A/B non-negative on Mobile Actions.
- **Optional-include head**: parked. Current decision is already ~89.5%
  per-decision against a 64% lexical baseline; a head must clear ~95% to halve
  the bucket. Build only if reranking under-delivers.
- Everything ships behind the standing rule: **every decode change is A/B'd
  against a suite it was not designed for** (the repetition blocker cost 51
  points for skipping this).

## 7. Explicitly excluded, with the measurement that excluded it

| idea | verdict |
|---|---|
| beam search / RL / best-of-N selection | pass@9 oracle = +5.0 → caps at 29.3 < 32.6 |
| DCCD / draft-then-constrain | projection tax measured at -1.3 and 0.0; nothing to recover |
| copy heads (span, pointer) | -30 and -16 here; literature: pointers hurt under GCD |
| 2D-RoPE | mechanism needs column alignment our format lacks |
| reasoning traces / structured plans | oracle: perfect selection buys +3; reasoning targets selection |
| global optional-skip prior | inclusion is catalog-dependent (33%–94%); constant trades suites |
| MAX_CALLS raise | measured 0.000 twice; cap never binds |
| curriculum ordering | 2026 evidence contradictory; not worth final-run variance |
| architecture changes (no-FFN/SAN, engram) | their own paper: 0.006 nats at matched params |
| param scaling | 6.4x data-limited; scaling worsens the binding ratio |
| sequence-level loss (graded MRT) | held in reserve as a post-train fine-tune if Seal lands within ~2 points |

## 8. Execution order

| # | step | cost | wall |
|---|---|---|---|
| 1 | download BitAgent, Dolci, argilla | 0 | 30m |
| 2 | convert + evidence-filter | 0 | 1h |
| 3 | n-gram contamination firewall (build + sweep all sources) | 0 | 1h |
| 4 | dedup across sources; report net yield | 0 | 30m |
| 5 | tokenizer rebuild (16k, digit singletons, no leakage) + retrain-tokenizer sanity tests | 0 | 45m |
| 6 | synth top-up if gaps remain (refusals, Seal dialect) | $0–50 | 0–2h |
| 7 | pack --seq-len 640; verify row count & mean length | 0 | 30m |
| 8 | provision pod; from-scratch 44M, 3 epochs, dev-split checkpoint selection | ~$3 | 6–9h |
| 9 | full scorecard on GPU (suites in parallel — learned that the slow way) | ~$1 | 1.5h |
| 10 | reranker if field-oracle gate passed; A/B vs MA | 0 | 2h |
| 11 | final eval + RESULTS.md + README update; BFCL reported cold | 0 | 1h |

Total: **~$4–54, roughly one day.** Terminate the pod when idle; both API keys
are still due rotation (exposed in an early transcript).

## 9. Kill criteria / honesty rails

- Any imported source that fails the n-gram firewall in bulk is dropped, not
  trimmed.
- If dev-loss selection and final checkpoint disagree by >0.5 points on the
  training-mix dev, investigate before trusting either.
- DroidCall keeps its caveat (their split is unseeded; ours is seeded —
  "same methodology, different rows").
- The comparison stays labelled **parameter-matched, not size-matched**
  (their numbers are 2-bit + 256-token window; ours fp32 + 640).
- If Seal-in lands 30.5–32.5: run the graded-cost sequence-level fine-tune
  (reserve lever) before conceding. If it lands under 29: write it up, the
  data hypothesis is falsified at this scale, and the honest conclusion is
  that 44M + ~10 tokens/param is short of this suite.
- BFCL is reported whatever it says; nothing in this plan tunes on it.
45M-class disclosure: v5 is 48.12M params (16,384-vocab tied embeddings add 3.67M over v4's 44.45M). Needle 2 is 45.0M. At their own 2-bit deployment standard v5 is 11.5MB against their 14MB. Reported as parameter-class-matched with exact counts, never rounded down.

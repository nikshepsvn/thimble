# Results — v1 checkpoint (2026-08-16)

44.1M params · bf16 · trained 3 epochs on 146k rows · single RTX 3090, ~1 hour

## Mobile Actions (google/mobile-actions eval split, 961 rows)

Ordered strict exact match — function names, call order, and every argument value.

| Model | Params | Accuracy | Name acc. | Non-empty | 1-call | 2-call |
|---|---|---|---|---|---|---|
| **tiny-toolcall (bf16)** | **44M** | **80.1** | **99.5** | **100.0** | **87.2** | **66.0** |
| LFM2.5 230M (f16, vLLM) | 230M | 69.1 | 93.0 | 98.9 | 76.1 | 55.0 |
| FunctionGemma 270M (f16, vLLM) | 270M | 64.0 | 87.3 | 98.9 | 73.0 | 46.2 |
| Needle 2 (CQ2-bit) | 45M | 63.7 | 98.3 | 99.4 | 71.3 | 48.4 |
| Apple FM (on-device) | ~3B | 57.6 | 94.2 | 95.5 | 64.5 | 43.8 |

Baseline rows are Cactus's published figures. Well-formed rate: 100.0 (structural).

## Our own suites (400 rows each)

| Suite | S2 lexical floor | heads-on | heads-off |
|---|---|---|---|
| held-out (in-distribution) | 21.2 | 100.0 | 100.0 |
| OOD (unseen tool schemas) | 16.8 | 80.2 | 80.0 |

S2 = training-free lexical selector (Looking Is Not Picking, arXiv 2606.16364).

## Stated asymmetries

Both directions, following Cactus's own practice:

- **Precision favors us.** We report bf16; Needle reports CQ2-bit measured end-to-end
  through its shipped C++ binary with a 256-token sliding window. 2-bit QAT is not
  yet done here, and quantization will cost some accuracy.
- **Domain adaptation favors us.** Mobile Actions' *train* split (8,693 rows, public,
  disjoint from eval) is in our mix ×3. Needle trained on its own device-action
  corpus. Our eval rows were firewalled throughout — but this is a domain-adapted
  result against a generalist, and should be read that way.
- **Deployment favors Needle.** They ship a 14MB binary that runs on an ESP32; we
  have a 168MB bf16 checkpoint and no engine.

## Not yet measured

DroidCall (200), Seal-Tools in/out (700/654), BFCL v4 single-turn (3,641; their
"overall" is the unweighted mean over 13 raw categories), ACEBench Normal.
No harness exists for these yet — no claims are made about them.

---

# v1 post-mortem: the Seal-Tools collapse (2026-08-16, evening)

The v1 checkpoint scored **1.0%** on Seal-Tools in-domain and **0.9%** out-of-domain,
against Needle 2's 32.6 / 28.7 — below even a training-free lexical baseline.

## Root cause

Our teacher prompt instructed the generator to invent tools with `snake_case` names.
It complied, for all 80,599 traces:

| Corpus | snake_case | camelCase |
|---|---|---|
| our v1 training data | 99.6% | 0.0% |
| Seal-Tools | 0.3% | 99.7% |

The model had never seen a camelCase function name. Worse, our snake_case names
tokenize as *single* tokens (`set_lights` → 1 token) while camelCase names fragment
into 6–12 subwords (`getPostmodernTheory` → `get P ost moder n The ory`), so name
scoring operated in an entirely different regime from the one it was trained on.

Two symptoms followed: the model refused 39% of rows outright, and among the ~5
candidates it did consider, it picked correctly 27% of the time — barely above the
20% chance rate.

## Fixes, measured independently

**1. camelCase-aware retrieval.** The retriever lowercased before tokenizing, so
`getPostmodernTheory` collapsed to one opaque token and matched nothing. Splitting on
case boundaries first took the lexical baseline's name accuracy from 26.6% → **90.7%**
on Seal-Tools in-domain. That figure is itself the finding: self-describing API
catalogs are nearly solved by string overlap, whereas device-verb catalogs
(Mobile Actions, lexical name-acc 66.5%) are not.

**2. Lexical prior, gated on two conditions.** Three variants, 150 rows each:

| Decode variant | Seal-Tools acc / name | Mobile Actions acc / name |
|---|---|---|
| ungated (v1 behaviour) | 3.3 / 24.0 | 78.0 / 100.0 |
| hard confidence gate | 8.0 / 48.0 | 78.0 / 100.0 |
| prior weighted by sharpness alone | 10.0 / 56.7 | 75.3 / 96.7 |
| **sharpness × model uncertainty** | **8.7 / 54.0** | **78.0 / 100.0** |

The prior earns weight only when it discriminates on that catalog *and* the model is
unsure. Weighting on sharpness alone overrode the model on catalogs it knows well and
cost 2.7 points on Mobile Actions; requiring both conditions recovers them.
Spurious refusals fell 39% → 23%.

**3. Round-2 data.** 33,736 new traces across four naming conventions
(snake_case, camelCase, PascalCase, dot.notation, SCREAMING_SNAKE) and 42 domains
outside device control. Corpus now 114,335 traces; training set 178,947 rows.

## The honest reading

Mobile Actions 80.1 was a domain-adapted result. Seal-Tools is what a genuinely
unfamiliar catalog does to this model, and the answer was: it collapses. The
architecture didn't fail — the corpus was a monoculture, and one line of a teacher
prompt caused it.

# Results — v2 checkpoint (2026-08-17)

44.1M params · bf16 · 178,947 training rows · one RTX 3090 · ~$43 total

All numbers below are full row counts with the final decoder. "Ungated" is the
model alone; "gated" adds a lexical prior that is weighted only when it
discriminates on that catalog *and* the model is uncertain. Both are reported
because if the prior carries a suite, that should be visible.

## Mobile Actions — google/mobile-actions eval split, 961 rows

| Model | Params | Accuracy | Name acc. | Non-empty | 1-call | 2-call |
|---|---|---|---|---|---|---|
| **Thimble (bf16)** | **44M** | **80.4** | **99.3** | **100.0** | **86.7** | **67.9** |
| LFM2.5 230M (f16) | 230M | 69.1 | 93.0 | 98.9 | 76.1 | 55.0 |
| FunctionGemma 270M (f16) | 270M | 64.0 | 87.3 | 98.9 | 73.0 | 46.2 |
| Needle 2 (CQ2-bit) | 45M | 63.7 | 98.3 | 99.4 | 71.3 | 48.4 |
| Apple FM (on-device) | ~3B | 57.6 | 94.2 | 95.5 | 64.5 | 43.8 |

Ungated, gated and heads-off all score 80.4 — identical to the decimal. On this
suite the lexical prior contributes nothing and the factorized name head
contributes nothing; the trunk plus grammar does the work.

## Seal-Tools — we lose, and by a lot

| Suite | Ungated | Gated | Needle 2 | Gap |
|---|---|---|---|---|
| in-domain (700) | 2.4 | 4.4 | **32.6** | −28.2 |
| out-of-domain (654) | 2.0 | 3.1 | **28.7** | −25.6 |

Name accuracy: 12.4 → 18.9 gated (in-domain), against Needle's 64.9. The gate
roughly doubles the score and is nowhere near enough.

**Why.** With names correct, only 8.3% of rows have every argument right. The
model copies argument spans well out of colloquial device requests and poorly out
of terse academic instructions ("Provide information on the swimming pattern of
manta rays" → `fish_type='manta ray'`). Round-2 data (34k traces, mixed naming
conventions, 42 non-device domains) moved in-domain from 1.0 to 4.4 without
closing it. This is a training-distribution gap, not a decoder gap.

## Our own splits, 400 rows each

| Split | Lexical floor | Ungated | Gated | Gated, heads off |
|---|---|---|---|---|
| in-distribution | 21.2 | — | 100.0 | 100.0 |
| unseen schemas (OOD) | 16.8 | 81.9 | **82.8** | 80.6 |

The OOD row is the name head's only positive result all night: +2.2 points over
the same trunk without it. Everywhere else the two are identical.

## Stated asymmetries

- **Precision favours us.** bf16 here; Needle reports CQ2-bit measured end-to-end
  through a shipped C++ binary with a 256-token sliding window.
- **Domain adaptation favours us on Mobile Actions.** Its public train split
  (8,693 rows, disjoint from eval) is in our mix. Seal-Tools shows what happens
  without that advantage.
- **Deployment favours Needle, enormously.** 14MB binary on an ESP32 versus our
  168MB bf16 checkpoint and no engine.

## Decoder disclosure

Reproducing these numbers requires: grammar-constrained decoding with schema-forced
argument keys, mid-generation re-retrieval between calls, value cap of 96 tokens,
periodic-repetition blocking, and the gated lexical prior (off by default in the
"ungated" column). The scorer is Needle's ordered strict exact match, unmodified.

## Not measured

DroidCall (200), BFCL v4 single-turn (3,641), ACEBench Normal. No harness exists
for these and no claims are made about them.

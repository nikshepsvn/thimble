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

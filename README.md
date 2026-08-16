# tiny-toolcall

A ~45M-parameter factorized tool-calling model, built to beat [Needle 2](https://github.com/cactus-compute/needle)
(Cactus, Aug 2026) on its own published tables at the same size class. MIT.

## Thesis

Needle 2's holes are not where its architecture is: name accuracy on Mobile Actions is
already 98.3%, yet 2-call accuracy is 48.4%. The failure is (a) the second call never
entering the decode, and (b) argument values — plus name *selection* on OOD suites
(DroidCall name-acc 36.5, Seal-Tools 58–65). "Looking Is Not Picking"
(arXiv 2606.16364) shows tool mis-selection is a **readout** failure: models attend to
the right tool and still pick wrong; readout-side fixes recover 59–91% of failures vs
≤23% for prompt-side. So we spend the parameter budget on the readout, not the mixer.

## Architecture

Deep-thin gated trunk (**d=448, 20 layers**, GQA 8/4, SwiGLU ×2.0, QK-norm, sandwich
RMSNorm, sigmoid gate after SDPA, tied embeddings) ≈ 44M — plus three factorized
capabilities:

1. **Retriever** — `retrieve(query, tools, emitted=...)`: DTDR-style
   (arXiv 2512.17052) refresh conditioned on the *partial plan*. Benchmarks are
   single-pass ordered strict match — there are no tool execution results — so the
   refresh happens after each **emitted call**, mid-generation.
2. **Name head** — bilinear readout scoring candidate tool-name spans in the prompt
   against the hidden state at the decision position. Must beat the training-free
   S2 attention/lexical baseline to earn its parameters; a same-trunk
   "heads-off" decode (LM logprob over the name trie) is the standing ablation.
3. **Grammar decoder** — the tokenizer keeps JSON structural chars as singleton
   tokens, so `grammar.py` force-feeds structure exactly and consults the model only
   at choice points: refuse-vs-call, name, optional-include, value content,
   stop-vs-continue. **Argument keys are grammar-forced from the schema** — the
   param-name-hallucination error class cannot occur. Prompts use a compact
   signature form (`- name (param=str! mode=enum(heat|cool)) desc`) because the
   singleton contract only pays off on the generated call.

Training: weighted CE matched to the observed error distribution (structure 1×,
keys 1.5×, names 2×, values 4×, **stop decision 6×** — the dominant 2-call failure is
emitting `]` after one call) + name-head CE. Muon on trunk 2D weights, AdamW on
embeddings/norms/heads. ~13% refusal rows, Hammer-style name-masking.

## Targets (Needle 2's own tables, ordered strict exact match)

| Suite | Needle 2 | Target |
|---|---|---|
| Mobile Actions (961) | 63.7 (2-call 48.4) | ≥70 (2-call ≥60) |
| DroidCall (200) | 17.0 | beat |
| Seal-Tools in / OOD | 32.6 / 28.7 | ≥ / ≥28 (no regression) |
| BFCL v4 single-turn, *their* aggregation | 42.6 | ≥55 |
| Well-formed | 93.4–99.4 | 100 (grammar) |

Note: Needle's "BFCL overall" is their own single-turn aggregation, **not** the
official v4 overall (which is 40% agentic). The harness must reproduce their math.

## Pipeline

```
ttc synth N [--split train|eval|ood]   local validated synth (free, no API)
ttc tok                                8k BPE on a sample of rendered text
ttc pack [--split S] [--n N]           uint16 arrays + name-decision sidecar
ttc train [--n N] [--epochs E]         weighted CE + name aux, Muon+AdamW
ttc eval [--ckpt NAME] [--split S]     S2 baseline vs heads-on vs heads-off
ttc overfit                            200-row loop proof
```

Data layers: (A) public corpora as schema/query seeds — never SFT'd raw
(Turnstile: raw OS data *hurt* a 0.6B); (B) teacher traces via OpenRouter
(DeepSeek V4 Flash volume + Gemini 3.7 Flash batch hard-20%, Exacto routing,
stepwise-validated, two-pass: 150k → smoke → failure-bucket-targeted remainder);
(C) official eval suites, read-only, firewalled in `data/eval/`.

## Kill criteria

- Smoke: well-formed ≪95% (impossible by construction — grammar), in-domain
  name-acc ≪90%, or heads-on fails to beat heads-off on 2-call + refuse → drop heads.
- Per-bench expectations: name head pays on DroidCall/Seal-Tools/BFCL;
  retriever-refresh + stop supervision pay on Mobile Actions 2-call.

## Budget

$80 OpenRouter cap + $80 RunPod cap (community 4090; kill when idle). Default path
to a scored checkpoint ≈ $40–100. 2-bit QAT (ParetoQ 90/10 split) is Phase 3 only.

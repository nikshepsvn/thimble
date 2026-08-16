# A 45M tool-caller that cannot emit invalid JSON

*Draft — Seal-Tools numbers marked TK are still running.*

Five days ago Cactus released [Needle 2](https://cactuscompute.com/needle), a 45M-parameter
tool-calling model that fits in 14MB and runs on a microcontroller. It scores 63.7% on
Google's Mobile Actions benchmark, ahead of models five times its size.

We spent one day and about $27 building a model in the same size class. On Mobile Actions
it scores **80.1%**.

This post is about why that gap exists, because it isn't scale and it isn't a better mixer.

## Tool calling is five decisions, not a generation problem

Most tool-calling models are language models that write JSON. Ours is a language model that
makes five decisions, with the JSON assembled around it by a compiler.

Look at what a tool call actually requires. Given a query and a catalog of typed functions,
something has to decide: (1) does any tool apply, (2) which one, (3) which optional arguments
are licensed by the query, (4) what values go in them, and (5) is another call needed. Every
other token in the output — the braces, the quotes, the commas, and critically *every argument
key* — is determined by the schema before the model runs.

A model that generates JSON as text spends capacity learning that `{` follows `[`. At 45M
parameters you cannot afford that. So we don't: a byte-level grammar compiled from the
declared schemas force-feeds all structure, and the model is consulted only at the five
choice points. Three failure modes become unreachable rather than unlikely:

- **Malformed JSON** — the structure is emitted by the compiler, not sampled.
- **Invented argument names** — keys come from the schema; the model never generates one.
- **Calls to tools that don't exist** — the name is chosen from a constrained candidate set.

Those aren't things we trained away. They're things the decoder cannot express.

## Where Needle actually loses

Needle 2's Mobile Actions profile is the interesting part of their paper: **98.3% name
accuracy but 63.7% overall, and 48.4% on two-call rows.** They almost never pick the wrong
tool. They lose on argument values and on second calls that never get made.

That second number is a mechanism, not a mystery. Their pipeline retrieves the top-5 tools
once, then generates the call array as tokens. If the second call needs a tool outside that
initial top-5, it was never a legal option — the model cannot recover. The fix is in a paper
they don't cite: [DTDR](https://arxiv.org/abs/2512.17052) conditions retrieval on the query
*plus the calls already emitted*. We re-run retrieval between array elements and swap the
grammar's legal name set mid-generation.

The other half is a stop decision. After one call, the model emits either `,` or `]` — a
single token that determines whether a two-call row can possibly be correct. We weight that
token 6× in the loss. Our two-call accuracy is **66.0%** against their 48.4%.

## The architecture

A deep-thin transformer trunk — d=448, 20 layers, GQA 8/4, SwiGLU, 8k BPE with tied
embeddings, QK-norm, sandwich normalization, gated attention — carrying three heads:

1. **A retriever** that re-ranks the catalog against the query and the partial plan.
2. **A name head**: a bilinear readout scoring the decision-position hidden state against
   each candidate's name span in the prompt. This comes from
   ["Looking Is Not Picking"](https://arxiv.org/abs/2606.16364), which showed models attend
   to the correct tool 80% of the time and still choose wrong — tool selection fails at the
   readout, not at perception. Readout-side interventions recover 59–91% of failures;
   prompt-side fixes recover at most 23%.
3. **A grammar-constrained argument decoder** where keys are forced and only values are free.

Notably, we did *not* clone Needle's Simple Attention Network. Their own paper prices the
architecture against a matched-parameter transformer at 0.006 nats. The mixer was never the
bottleneck.

## Results

Mobile Actions, google/mobile-actions eval split, all 961 rows, ordered strict exact match:

| Model | Params | Accuracy | Name acc. | 1-call | 2-call |
|---|---|---|---|---|---|
| **ours (bf16)** | **44M** | **80.1** | **99.5** | **87.2** | **66.0** |
| LFM2.5 230M (f16) | 230M | 69.1 | 93.0 | 76.1 | 55.0 |
| FunctionGemma 270M (f16) | 270M | 64.0 | 87.3 | 73.0 | 46.2 |
| Needle 2 (CQ2-bit) | 45M | 63.7 | 98.3 | 71.3 | 48.4 |
| Apple FM (on-device) | ~3B | 57.6 | 94.2 | 64.5 | 43.8 |

Seal-Tools in-domain: TK vs 32.6. Out-of-domain: TK vs 28.7.

## What is unfair about this comparison

Cactus states their asymmetries openly, so we will too.

**Precision favors us.** We report bf16. Needle reports CQ2-bit, measured end-to-end through
a shipped C++ binary with a 256-token sliding window and cache eviction on. Quantizing to
2 bits will cost us accuracy we have not yet paid.

**Domain adaptation favors us.** Mobile Actions' train split — 8,693 public rows, disjoint
from eval — is in our training mix. Needle trained on its own device-action corpus. Our eval
rows were firewalled throughout and never seen, so the number is clean; but this is a
domain-adapted model measured against a generalist, and the Seal-Tools result is the better
test of whether the architecture generalizes.

**Deployment favors Needle, enormously.** They ship a dependency-free 14MB binary that runs
on an ESP32 with a 28MB RAM ceiling. We have a 168MB bf16 checkpoint and no engine. Their
2-bit QAT is trained from pretraining onward; ours doesn't exist. On the axis they actually
optimized for, we are not competitive.

## What it cost

One RTX 3090 at $0.22/hour for about an hour of training. 80,599 teacher-generated traces
from DeepSeek V4 Flash for $25. Total under $30 and one working day, which is the part that
should be surprising: the expensive thing was never the compute.

## What we got wrong along the way

The first checkpoint scored **0.0%** on Mobile Actions. Not low — zero. It refused half the
rows outright.

The cause wasn't capability. It was distribution: we had trained on catalogs of 3–9 tools
and Mobile Actions presents 10–16, with a date/time preamble on every query that our data
never contained. The model had never seen the shape of the question. Adding catalog
augmentation and preambles to the training mix moved it from 0.0 to 80.1.

Three bugs found by probing intermediate checkpoints, each of which would have silently
corrupted the headline:

- **Value truncation at 24 tokens** — email bodies were being cut mid-sentence, failing
  exact match on rows the model had otherwise answered perfectly.
- **A device mismatch** in the head/LM ensemble that crashed every constrained decode.
- **A tokenizer alignment break** on non-ASCII characters, where `<unk>` — a five-character
  token string standing in for one character — silently desynchronized the loss tags.

And one bug we nearly shipped: Mobile Actions' own schema advertises datetimes as
`YYYY-MM-DDTHH:MM:SS`, while its gold values use a space separator. We had written a grammar
template from the schema text. Checking the gold data instead of trusting the prose caught it.

## What's still open

The name head is on probation. Against a same-trunk ablation that just emits JSON, it wins
by 1.5 points on out-of-distribution data with a small training set — and ties once the
training set is large. The plan said we drop the heads if they don't beat the ablation. That
verdict is still pending on the harder suites.

DroidCall, BFCL v4, and ACEBench have no harness yet, and no claims are made about them.

MIT licensed. Code, data pipeline, and eval harness are in the repo.

# Launch kit (not for the repo's front page — delete or keep, your call)

## Hacker News

**Title options (HN hates hype; specificity wins):**
1. `Show HN: Thimble – a 48M tool-calling model that beats Needle 2 on 3 of its 5 benchmarks with 150x less data`
2. `Show HN: We beat a 153B-token model on its own benchmark for $260`
3. `Show HN: Thimble – grammar-constrained tool calling at 48M params (MIT)`

Option 1 is the safest; 2 is the punchiest but invites "cherry-picking" pushback —
survivable because the receipts are genuinely good.

**First comment (post immediately, sets the thread's tone):**

> Author here. Quick honest summary since benchmark posts deserve skepticism:
>
> - Their metric (ordered strict exact match), their published numbers, unmodified.
> - The Seal-Tools win is +0.5 on 700 rows — within sampling noise, and the README
>   says so. The other two wins are +22.6 and 3.1x.
> - We lose two tables: Seal out-of-domain by 0.6, and BFCL badly (23.5 vs 42.6) —
>   that one is their 153B-token data moat and we quantify exactly why we can't
>   cross it at 1B tokens.
> - The repo publishes every negative result (13 of them, including a
>   champion-selection protocol failure where our pre-registered dev metric picked
>   the wrong model — both models' numbers are published).
>
> The two ideas that did the work: (1) a grammar that force-feeds all JSON
> structure so the model only makes five decisions per call, and (2) synthesizing
> training data directly against the previous model's diagnosed failure buckets,
> then annealing it into the LR-decay phase — a controlled twin showed the same
> data fed from scratch scores 4.7 points worse.
>
> Total cost ~$260. Happy to answer anything, including the embarrassing parts.

## X/Twitter thread

**1/** We built a 48M-param tool-calling model that beats Needle 2 — a 45M model
trained on 153B tokens — on 3 of its 5 published benchmarks. Including Seal-Tools,
the one it's named after.

Ours saw ~1B tokens. 150x less. Total cost: $260. [chart image]

**2/** The trick isn't scale, it's a division of labor: a grammar compiled from the
tool schemas emits all the JSON — braces, quotes, every argument key. The model
only answers five questions: refuse? which tool? include this optional? what
value? another call?

Malformed JSON isn't unlikely. It's unreachable.

**3/** The data was built like a bug tracker: diagnose the failure buckets of the
last model (66 spurious optionals, 35 wrong-slot bindings, 30 date-format misses
out of 193 failing calls), synthesize corrective examples for each, verify the fix
causally mid-training (+3.3 pts at constant LR).

**4/** Coolest finding: the *same* corrective data fed two ways —
- from scratch: 28.4 (dilutes into the average)
- annealed into the LR-decay phase of a continued run: 33.1 (concentrates where
  behavior crystallizes)

The decay phase is where your best data belongs.

**5/** Most embarrassing finding, published anyway: our pre-registered model
selector (held-out loss on the training mix) picked the *wrong* twin — a
general-mix dev metric is structurally biased against a model that annealed
toward the eval distribution. Both models' full numbers are in the repo.

**6/** What we couldn't beat: BFCL (23.5 vs 42.6). That's what 153B tokens of
breadth buys and 1B of concentrate can't fake — Java/JS schema dialects,
paraphrase-distance values. We quantify the boundary instead of excusing it.

**7/** Everything is MIT: weights on HF, code + 13 published negative results on
GitHub, and a 30-second demo. Built by one person + an AI assistant in a week of
evenings.

[links]

## Reddit (r/LocalLLaMA)

Title: `Thimble: 48M tool-calling model that beats Needle 2 on 3/5 of its own benchmarks (MIT, ~1B training tokens, $260)`
Body: results table + the anneal-vs-scratch finding + demo output + links. This
crowd cares most about: runs-on-anything size, MIT license, the grammar guarantee,
and honest negative results. Lead with the demo transcript.

## Release-day order

1. Rotate OpenRouter + RunPod keys; optionally reissue the HF token.
2. GitHub repo -> public. Set social preview: Settings -> General -> Social preview,
   upload assets/social_card.png.
3. HF model -> public. (Space needs HF PRO if wanted: `hf_space/` is deploy-ready.)
4. Publish the blog post (BLOG.md).
5. HN post (weekday morning US time is the usual advice), first comment immediately.
6. X thread + r/LocalLLaMA an hour later, linking the HN thread if it has traction.

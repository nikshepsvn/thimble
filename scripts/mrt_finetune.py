"""RESERVE LEVER — graded-reward fine-tune (RLOO over sampled decodes).

Fires only if the champion lands inside the 29-32.5 band on Seal-in.

Mechanism, and why pass@k does not bound it: for each training row we draw k
constrained samples and score each with PARTIAL credit r = fraction of gold
calls matched exactly at their positions. The REINFORCE gradient with a
leave-one-out baseline pushes UP samples above their siblings and DOWN samples
below — so a pair like {2/3 correct, 1/3 correct} carries signal even though
neither is a full success. Whole-sample pass@k only counted full successes;
the differential between near-misses is exactly the conjunction signal it
missed. Gold-CE is mixed at alpha=0.3 (Edunov et al.: pure sequence-level
training underperforms the weighted combination).

Training rows are drawn from the PACKED TRAIN rows only (never dev, never
eval); multi-call rows are preferred since the conjunction is the target.

EXPERIMENTAL. Smoke with --n 200 before a real run. Warm-starts the champion;
LR scaled to 0.1x; a short pass (1 epoch over the sampled subset) is the point
— this is calibration, not capability training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_toolcall.data import name_spans_in_prompt, normalize_example
from tiny_toolcall.grammar import constrained_decode
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.render import prompt_text, render_example
from tiny_toolcall.schema import canon_calls, dumps_calls
from tiny_toolcall.tokenizer import BOS, BPETokenizer
from tiny_toolcall.train import split_params

ROOT = Path(__file__).resolve().parents[1]


def reward(pred: list[dict], gold: list[dict]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    hits = sum(1 for i, g in enumerate(gold) if i < len(pred) and pred[i] == g)
    return hits / len(gold) - 0.1 * max(0, len(pred) - len(gold))


def seq_logprob(model, tok, prompt: str, calls: list[dict], dev) -> torch.Tensor:
    """Differentiable log P(call string | prompt), summed over call tokens."""
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(dumps_calls(calls))
    ids = torch.tensor([[BOS] + p_ids + c_ids], device=dev)
    logits, _ = model(ids)
    lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = ids[0, 1:]
    start = len(p_ids)  # first call token position in tgt space
    return lp[torch.arange(start, len(tgt)), tgt[start:]].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000, help="training rows to sample")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.3, help="gold-CE mix weight")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    rows = [json.loads(l) for l in
            (ROOT / "data/packed/train/rows.jsonl").read_text().splitlines()[: 400000] if l.strip()]
    rng = random.Random(a.seed)
    multi = [r for r in rows if len(r.get("answers", [])) >= 2]
    single = [r for r in rows if len(r.get("answers", [])) < 2]
    pick = rng.sample(multi, min(int(a.n * 0.7), len(multi))) + \
           rng.sample(single, min(a.n - int(a.n * 0.7), len(single)))
    rng.shuffle(pick)
    print(f"RLOO fine-tune: {len(pick)} rows (70% multi-call), k={a.k}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    t0 = time.time()
    used = skipped = 0
    for i, raw in enumerate(pick):
        ex = normalize_example(raw)
        pr = prompt_text(ex["query"], ex["tools"])
        pid = tok.encode(pr)
        if len(pid) > 600:
            skipped += 1
            continue
        s0 = name_spans_in_prompt(tok, pr, pid, [t["name"] for t in ex["tools"]])
        spans = {nm: (x + 1, y + 1) for nm, (x, y) in s0.items()}
        gold = canon_calls(ex["answers"])
        model.eval()
        with torch.no_grad():
            samples = [canon_calls(constrained_decode(
                model, tok, pr, ex["query"], ex["tools"], dev,
                name_spans=spans, gated=True, temp=a.temp)) for _ in range(a.k)]
        rs = [reward(s, gold) for s in samples]
        if max(rs) - min(rs) < 1e-6:
            skipped += 1     # no differential -> no RLOO signal; gold-CE only
            continue
        model.train()
        loss = model.embed.weight.new_zeros(())
        for j, (s, r) in enumerate(zip(samples, rs)):
            baseline = (sum(rs) - r) / (len(rs) - 1)          # leave-one-out
            adv = r - baseline
            if abs(adv) < 1e-6 or not s:
                continue
            loss = loss - adv * seq_logprob(model, tok, pr, s, dev)
        loss = loss / a.k + a.alpha * (-seq_logprob(model, tok, pr, gold, dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        used += 1
        if used % 50 == 0:
            print(f"{used} updates ({skipped} skipped) loss={loss.item():.3f} "
                  f"mean_r={sum(rs)/len(rs):.2f} ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "checkpoints" / f"{a.out}.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__}, out)
    print(f"saved {out}  ({used} updates, {skipped} skipped)")


if __name__ == "__main__":
    main()

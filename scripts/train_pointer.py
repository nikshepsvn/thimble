"""Train the pointer/copy head with the trunk frozen.

Supervision is derived, not annotated: for every string argument whose gold value
appears as a contiguous TOKEN subsequence of the prompt, the (start, end) indices
are the label. Values that are transformed rather than copied — datetime
conversions, morphological variants — simply produce no training signal and fall
back to free generation at decode time.

Only ptr_start / ptr_end receive gradients (~0.4M of 44.45M parameters), so no
existing capability can regress.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_toolcall.data import normalize_example
from tiny_toolcall.model import Config, ToolTransformer
from tiny_toolcall.render import prompt_text
from tiny_toolcall.schema import dumps_calls
from tiny_toolcall.tokenizer import BOS, BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def find_sub(hay: list[int], needle: list[int]) -> int:
    n = len(needle)
    if not n or n > len(hay):
        return -1
    first = needle[0]
    for i in range(len(hay) - n + 1):
        if hay[i] == first and hay[i : i + n] == needle:
            return i
    return -1


def build_examples(rows, tok, seq_len: int, limit: int):
    """-> list of (ids, prompt_len, decision_pos, start, end)"""
    out = []
    for ex in rows:
        ex = normalize_example(ex)
        prompt = prompt_text(ex["query"], ex["tools"])
        call = dumps_calls(ex["answers"])
        p_ids, c_ids = tok.encode(prompt), tok.encode(call)
        if 1 + len(p_ids) + len(c_ids) > seq_len:
            continue
        ids = [BOS] + p_ids + c_ids
        base = 1 + len(p_ids)
        # walk the call string, locating each string value's token range
        cursor = 0
        for c in ex["answers"]:
            for k in sorted(c["arguments"]):
                v = c["arguments"][k]
                if not isinstance(v, str) or len(v) < 2:
                    continue
                marker = f'"{k}":"'
                at = call.find(marker + v, cursor)
                if at < 0:
                    continue
                cursor = at + len(marker) + len(v)
                # token index in the call where the value begins
                pre = tok.encode(call[: at + len(marker)])
                val_ids = tok.encode(v)
                if not val_ids:
                    continue
                src = find_sub(p_ids, val_ids)
                if src < 0:
                    continue  # transformed value, not copied — no signal
                dec_pos = base + len(pre) - 1
                if dec_pos <= 0 or dec_pos >= len(ids):
                    continue
                out.append((ids, len(p_ids), dec_pos, src, src + len(val_ids) - 1))
                if len(out) >= limit:
                    return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="data/packed/train/rows.jsonl")
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    blob = torch.load(ROOT / "checkpoints" / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = ToolTransformer(Config(**blob["cfg"]))
    missing = model.load_state_dict(blob["model"], strict=False)
    print("fresh params:", [k for k in missing.missing_keys], flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("ptr_")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M of "
          f"{model.count_params()/1e6:.2f}M", flush=True)

    src = [json.loads(l) for l in (ROOT / a.rows).read_text().splitlines() if l.strip()]
    random.Random(0).shuffle(src)
    data = build_examples(src, tok, seq_len=640, limit=a.limit)
    print(f"pointer supervision: {len(data)} spans from {len(src)} rows", flush=True)
    if not data:
        raise SystemExit("no supervision extracted")

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
    model.train()
    step = 0
    t0 = time.time()
    for ep in range(a.epochs):
        random.Random(ep).shuffle(data)
        for ids, plen, dpos, gs, ge in data:
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                _, hidden = model(x, need_logits=False)
            hidden = hidden.detach()
            s_log, e_log = model.pointer_scores(hidden, dpos, plen)
            loss = (F.cross_entropy(s_log[None], torch.tensor([gs], device=dev))
                    + F.cross_entropy(e_log[None], torch.tensor([ge], device=dev)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 500 == 0:
                with torch.no_grad():
                    ok = int(s_log.argmax() == gs) + int(e_log.argmax() == ge)
                print(f"ep{ep} step{step} loss={loss.item():.3f} hit={ok}/2 "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)
    out = ROOT / "checkpoints" / f"{a.out}.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__}, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()

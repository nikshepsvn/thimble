"""ttc: synth / tok / pack / train / eval / overfit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
TOK_PATH = DATA / "tokenizer.json"


def load_cfg() -> dict:
    import yaml  # optional; fall back to defaults if missing

    p = ROOT / "configs" / "45m.yaml"
    return yaml.safe_load(p.read_text())


def _cfg() -> dict:
    try:
        return load_cfg()
    except ModuleNotFoundError:
        return {}


def cmd_synth(args) -> None:
    from tiny_toolcall.synth import generate

    out = DATA / "seeds" / f"local_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(args.n, seed=args.seed, split=args.split)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"wrote {len(rows)} -> {out}")


def _train_rows() -> list[dict]:
    """Training mix: local synth + teacher traces when present."""
    rows = _read_rows(DATA / "seeds" / "local_train.jsonl")
    teacher = DATA / "synth" / "teacher.jsonl"
    if teacher.exists():
        trows = _read_rows(teacher)
        rows = rows + trows
        print(f"mix: {len(rows) - len(trows)} local + {len(trows)} teacher")
    return rows


def cmd_tok(args) -> None:
    from tiny_toolcall.render import render_example
    from tiny_toolcall.tokenizer import train_bpe

    rows = _train_rows()
    rng = random.Random(0)
    sample = rng.sample(rows, min(args.sample, len(rows)))
    texts = []
    for ex in sample:
        prompt, call, _ = render_example(ex)
        texts.append(prompt + call)
    tok = train_bpe(texts, vocab_size=args.vocab)
    tok.save(TOK_PATH)
    print(f"vocab={tok.vocab_size} merges={len(tok.merges)} -> {TOK_PATH}")


def cmd_pack(args) -> None:
    import numpy as np

    from tiny_toolcall.data import pack_examples, save_packed
    from tiny_toolcall.tokenizer import BPETokenizer

    tok = BPETokenizer.load(TOK_PATH)
    rows = _train_rows() if args.split == "train" else _read_rows(DATA / "seeds" / f"local_{args.split}.jsonl")
    random.Random(7).shuffle(rows)  # interleave sources
    if args.n:
        rows = rows[: args.n]
    ids, tags, dec, kept = pack_examples(rows, tok, seq_len=args.seq_len)
    out = DATA / "packed" / args.split
    save_packed(out, ids, tags, dec)
    # the exact rows packed, in order — evals must read this, not the seeds file
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept))
    print(f"packed {ids.shape} -> {out}  (mean real len {int((ids != 0).sum(1).mean())})")


def cmd_train(args) -> None:
    from tiny_toolcall.data import load_packed
    from tiny_toolcall.model import build
    from tiny_toolcall.tokenizer import BPETokenizer
    from tiny_toolcall.train import pick_device, train

    cfg = _cfg()
    tok = BPETokenizer.load(TOK_PATH)
    ids, tags, dec = load_packed(DATA / "packed" / args.split)
    if args.n:
        ids, tags, dec = ids[: args.n], tags[: args.n], dec[: args.n]
    model = build(tok.vocab_size, **{k: v for k, v in cfg.get("model", {}).items() if k != "vocab_size"})
    print(f"params: {model.count_params()/1e6:.2f}M  device: {pick_device()}")
    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    stats = train(model, ids, tags, dec, cfg, save_path=CKPT / f"{args.name}.pt")
    print("final:", stats)


def cmd_eval(args) -> None:
    import torch

    from tiny_toolcall.eval import make_model_predictor, predict_s2, score_predictor
    from tiny_toolcall.model import Config, ToolTransformer
    from tiny_toolcall.tokenizer import BPETokenizer
    from tiny_toolcall.train import pick_device

    rows = _read_rows(DATA / "seeds" / f"local_{args.split}.jsonl")
    if args.n:
        rows = rows[: args.n]
    print(f"S2 baseline ({args.split}, n={len(rows)}):")
    print(_fmt(score_predictor(rows, predict_s2)))
    if args.ckpt:
        tok = BPETokenizer.load(TOK_PATH)
        blob = torch.load(CKPT / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
        model = ToolTransformer(Config(**blob["cfg"]))
        model.load_state_dict(blob["model"])
        device = pick_device()
        model.to(device).eval()
        for heads in (True, False):
            label = "heads-on" if heads else "heads-off (ablation)"
            pred = make_model_predictor(model, tok, device, use_name_head=heads)
            print(f"model {label}:")
            print(_fmt(score_predictor(rows, pred)))


def cmd_official(args) -> None:
    import torch

    from tiny_toolcall.eval import make_model_predictor, predict_s2, score_predictor
    from tiny_toolcall.model import Config, ToolTransformer
    from tiny_toolcall.official import mobile_actions_rows
    from tiny_toolcall.tokenizer import BPETokenizer
    from tiny_toolcall.train import pick_device

    rows = mobile_actions_rows(DATA / "eval" / "mobile_actions_eval.jsonl")
    if args.n:
        rows = rows[: args.n]
    print(f"Mobile Actions eval, n={len(rows)}  (Needle 2: acc 63.7, name 98.3, 1-call 71.3, 2-call 48.4)")
    print("S2 baseline:")
    print(_fmt(score_predictor(rows, predict_s2)))
    if args.ckpt:
        tok = BPETokenizer.load(TOK_PATH)
        blob = torch.load(CKPT / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
        model = ToolTransformer(Config(**blob["cfg"]))
        model.load_state_dict(blob["model"])
        device = pick_device()
        model.to(device).eval()
        for heads in (True, False):
            pred = make_model_predictor(model, tok, device, use_name_head=heads)
            print("model " + ("heads-on:" if heads else "heads-off:"))
            print(_fmt(score_predictor(rows, pred)))


def cmd_teacher(args) -> None:
    import asyncio

    from tiny_toolcall.teacher import synth_teacher

    out = DATA / "synth" / "teacher.jsonl"
    stats = asyncio.run(synth_teacher(args.n, out, concurrency=args.concurrency, seed=args.seed))
    print(stats)


def cmd_overfit(args) -> None:
    """Prove the loop: tiny model, 200 rows, then decode those same rows."""
    args.n = 200
    args.split = "train"
    args.epochs = args.epochs or 30
    args.name = "overfit"
    cmd_train(args)
    args.ckpt = "overfit"
    cmd_eval(args)


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _fmt(s: dict) -> str:
    keys = ["accuracy", "name_acc", "well_formed", "non_empty", "one_call", "two_plus", "refuse"]
    return "  " + "  ".join(f"{k}={s.get(k, 0):.3f}" for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(prog="ttc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth")
    p.add_argument("n", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="train")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("tok")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--sample", type=int, default=8000)
    p.set_defaults(fn=cmd_tok)

    p = sub.add_parser("pack")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=512)
    p.set_defaults(fn=cmd_pack)

    p = sub.add_parser("train")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--name", default="sft")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--split", default="eval")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--ckpt", default="")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("overfit")
    p.add_argument("--epochs", type=int, default=0)
    p.set_defaults(fn=cmd_overfit)

    p = sub.add_parser("teacher")
    p.add_argument("n", type=int)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_teacher)

    p = sub.add_parser("official")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--ckpt", default="")
    p.set_defaults(fn=cmd_official)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

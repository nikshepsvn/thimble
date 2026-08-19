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
    """Training mix: local synth + teacher traces + official train seeds,
    with MA-shaped augmentation."""
    # local templates saturate fast — cap them so diverse data dominates;
    # official MA-train is the scarcest, highest-value signal — repeat it 3x
    rows = _read_rows(DATA / "seeds" / "local_train.jsonl")[:40000]
    counts = {"local": len(rows)}
    teacher = DATA / "synth" / "teacher.jsonl"
    if teacher.exists():
        extra = _read_rows(teacher)
        counts["teacher"] = len(extra)
        rows += extra
    # v6 targeted synth (omission / entity-distractor / date-canonicalization
    # focus modes, aimed at the measured v5 failure buckets) — x3: it is the
    # corrective signal this run exists to add
    teacher_v6 = DATA / "synth" / "teacher_v6.jsonl"
    if teacher_v6.exists():
        extra = _read_rows(teacher_v6)
        counts["teacher_v6_x3"] = len(extra) * 3
        rows += extra * 3
    # public benchmark TRAIN splits, repeated: they are the scarcest and most
    # valuable signal we have, and each one is the same distribution as an eval
    # suite we are measured on (train/eval firewall verified at conversion time)
    for name, fname, rep in (("mobile_actions_x3", "official_train.jsonl", 3),
                             ("seal_tools_x6", "seal_train.jsonl", 6),
                             # x6, up from x3: seal_train is a perfect distributional
                             # twin of the eval set (69/99.8/35.1/13.6 vs 70/99.7/35.9/14.0
                             # on 3+calls/camelCase/opt-inclusion/numerics) and at x3 it
                             # was 5% of a mix whose other 95% teaches 62-95% optional
                             # inclusion — the exact over-inclusion error that is 28.4%
                             # of our per-call failures. The model was doing what the
                             # aggregate taught it.
                             ("droidcall_x3", "droidcall_train_heldout.jsonl", 3),  # 200 rows held out for eval
                             ("xlam", "xlam.jsonl", 1),
                             ("toolace_x2", "toolace.jsonl", 2),
                             # typed-Python-signature dialect; 27% of its rows are
                             # parallel/multiple calls, the shape we score worst on
                             ("dria", "dria.jsonl", 1),
                             ("hermes_x3", "hermes.jsonl", 3),
                             # v5 imports — post firewall + cross-source dedup.
                             # dolci is the one genuine find of the bulk-import
                             # round: 120,511 truly new rows. bitagent (44 unique
                             # of 551k) and argilla-apigen (0 unique of 109k)
                             # turned out to be re-hosts of corpora already in
                             # the mix and are deliberately absent.
                             ("dolci", "dolci.jsonl", 1),
                             ("glaive_sg", "glaive_sg.jsonl", 1),
                             ("hermes_reason", "hermes_reason.jsonl", 1),
                             ("qwen_tc", "qwen_tc.jsonl", 1),
                             ("fc_unfiltered", "fc_unfiltered.jsonl", 1),
                             ("dria_steps", "dria_steps.jsonl", 1),
                             # rows (already in the mix x1 via their sources)
                             # where a call omits >=1 available optional while
                             # using others — selective omission is the v5
                             # error bucket the aggregate mix under-teaches;
                             # +2 reps here makes them x3 effective
                             ("omission_x2", "omission_exemplars.jsonl", 2)):
        f = DATA / "seeds" / fname
        if f.exists():
            extra = _read_rows(f)
            counts[name] = len(extra) * rep
            rows += extra * rep
    print("mix:", counts)
    return _augment(_firewall(rows))


def _firewall(rows: list[dict]) -> list[dict]:
    """Drop any training row whose query matches an eval query.

    Contamination is checked here rather than trusted from the source splits:
    one teacher-generated query ("Turn on the light in the kitchen") collided
    with a Seal-Tools OOD row by coincidence, which is exactly the kind of thing
    that only shows up if you look.
    """
    import re

    f = DATA / "eval_queries.txt"
    if not f.exists():
        print("WARNING: no eval-query firewall list; run scripts/build_firewall.py")
        return rows
    banned = set(f.read_text().splitlines())
    norm = lambda q: re.sub(r"\W+", " ", q.lower()).strip()
    keep = [r for r in rows if norm(r["query"]) not in banned]
    if len(keep) != len(rows):
        print(f"firewall: dropped {len(rows) - len(keep)} contaminated rows")
    return keep


def _augment(rows: list[dict]) -> list[dict]:
    """Close the observed eval-distribution gaps without API cost:
    (a) big catalogs — official suites show 10-16 tools, training peaked at 9;
        top up ~40% of rows with distractor schemas pooled across the mix
    (b) context preamble — Mobile Actions prepends date/time lines to every
        query; zero-shot, those made the model refuse half the rows
    """
    rng = random.Random(11)
    pool: list[dict] = []
    for r in rows:
        pool.extend(r["tools"])
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = []
    for r in rows:
        r = dict(r)
        if pool and rng.random() < 0.4:
            have = {t["name"] for t in r["tools"]}
            target = rng.randint(10, 16)
            tools = list(r["tools"])
            for _ in range(40):
                if len(tools) >= target:
                    break
                cand = pool[rng.randrange(len(pool))]
                if cand["name"] not in have:
                    tools.append(cand)
                    have.add(cand["name"])
            rng.shuffle(tools)
            r["tools"] = tools
        if rng.random() < 0.35 and not r["query"].startswith("Current date"):
            # official MA rows already carry their real preamble — don't double it
            dt = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}"
            r["query"] = (
                f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {dt} "
                f"Day of week is {rng.choice(days)}\n{r['query']}"
            )
        out.append(r)
    return out


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
    if getattr(args, "init", ""):
        # warm start: the tokenizer is unchanged, so weights transfer exactly
        import torch

        blob = torch.load(CKPT / f"{args.init}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        print(f"warm-started from {args.init}.pt")
    print(f"params: {model.count_params()/1e6:.2f}M  device: {pick_device()}")
    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if getattr(args, "lr_scale", 1.0) != 1.0:
        cfg.setdefault("train", {})["lr_scale"] = args.lr_scale
    # RFT-style forced-token down-weighting (SHAD/RFT, ACL Findings 2025): the
    # grammar force-feeds structure and keys at decode, and they consume 46.4%
    # of the weighted loss budget. The twin run sets both to 0.1.
    if getattr(args, "w_structure", None) is not None:
        cfg.setdefault("loss", {})["structure"] = args.w_structure
    if getattr(args, "w_keys", None) is not None:
        cfg.setdefault("loss", {})["keys"] = args.w_keys
    if getattr(args, "anneal_data", ""):
        cfg.setdefault("train", {})["anneal_dir"] = str(DATA / "packed" / args.anneal_data)
    if getattr(args, "no_warmup", False):
        cfg.setdefault("train", {})["no_warmup"] = True
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
    """Tolerant JSONL reader: a file being APPENDED to (teacher synth writes
    while tok/pack read) always risks one partially-written trailing line.
    This exact race corrupted teacher.jsonl once already; skip-and-count is the
    correct behaviour, not crashing half an hour into a mix load."""
    rows, bad = [], 0
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  _read_rows: skipped {bad} undecodable line(s) in {path.name}")
    return rows


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
    p.add_argument("--init", default="", help="warm-start from this checkpoint name")
    p.add_argument("--lr-scale", type=float, default=1.0, help="scale LRs (use <1 when warm-starting)")
    p.add_argument("--anneal-data", default="", help="packed dir name used for the decay phase (data annealing)")
    p.add_argument("--no-warmup", action="store_true", help="skip LR warmup (continued runs from a plateau checkpoint)")
    p.add_argument("--w-structure", type=float, default=None, help="loss weight override for forced structural tokens")
    p.add_argument("--w-keys", type=float, default=None, help="loss weight override for forced key tokens")
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
